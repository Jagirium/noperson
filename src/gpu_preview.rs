//! GPU-native preview transport shared by the realtime and editor frontends.
//!
//! The state machine is intentionally independent from a concrete graphics
//! backend. Linux attaches CUDA/Vulkan resources to these slots; other
//! platforms can keep using the CPU fallback until their native interop lands.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

const FREE: u8 = 0;
const WRITING: u8 = 1;
const READY: u8 = 2;
const READING: u8 = 3;
const WGPU_COPY_ALIGNMENT: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewGeometry {
    width: u32,
    height: u32,
    row_bytes: u32,
    buffer_size: u64,
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::File;
    use std::mem::ManuallyDrop;
    use std::os::fd::FromRawFd;
    use std::sync::{Arc, Mutex};

    use super::{PreviewGeometry, PreviewRingState};
    use crate::backend::cuda::{
        DeviceSlice, SyncOnDrop, driver_result as result, driver_sys as sys,
    };
    use crate::backend::{Buffer, ComputeEvent, ComputeStream, DevicePtrMut};
    use anyhow::{Context as _, anyhow};
    use ash::vk;

    const SLOT_COUNT: usize = 3;

    struct CudaInteropBuffer {
        external_memory: sys::CUexternalMemory,
        device_ptr: sys::CUdeviceptr,
        len: usize,
        stream: Arc<ComputeStream>,
        completion: ComputeEvent,
        written: bool,
    }

    // The imported allocation is confined to the CUDA worker. The mutex in
    // `PreviewSlot` is the ownership boundary when the bridge is torn down.
    unsafe impl Send for CudaInteropBuffer {}

    impl DeviceSlice<u8> for CudaInteropBuffer {
        fn len(&self) -> usize {
            self.len
        }

        fn stream(&self) -> &Arc<ComputeStream> {
            &self.stream
        }
    }

    impl DevicePtrMut<u8> for CudaInteropBuffer {
        fn device_ptr_mut<'a>(
            &'a mut self,
            stream: &'a ComputeStream,
        ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
            if self.written {
                stream
                    .wait(&self.completion)
                    .expect("preview CUDA event belongs to the same context");
            }
            self.written = true;
            (
                self.device_ptr,
                SyncOnDrop::Record(Some((&self.completion, stream))),
            )
        }
    }

    impl Drop for CudaInteropBuffer {
        fn drop(&mut self) {
            let context = self.stream.context();
            let _ = context.bind_to_thread();
            let _ = self.stream.synchronize();
            let _ = unsafe { result::memory_free(self.device_ptr) };
            let _ =
                unsafe { result::external_memory::destroy_external_memory(self.external_memory) };
        }
    }

    struct PreviewSlot {
        buffer: wgpu::Buffer,
        import_file: Mutex<Option<File>>,
        cuda: Mutex<Option<CudaInteropBuffer>>,
    }

    pub struct LinuxPreviewBridge {
        geometry: PreviewGeometry,
        ring: Arc<PreviewRingState>,
        slots: Box<[PreviewSlot]>,
        render_state: egui_wgpu::RenderState,
        texture: wgpu::Texture,
        texture_id: egui::TextureId,
    }

    pub struct LinuxPreviewWrite<'a> {
        bridge: &'a LinuxPreviewBridge,
        slot: Option<usize>,
    }

    impl LinuxPreviewWrite<'_> {
        pub fn commit(mut self) {
            if let Some(slot) = self.slot.take() {
                self.bridge.ring.publish(slot);
            }
        }
    }

    impl Drop for LinuxPreviewWrite<'_> {
        fn drop(&mut self) {
            if let Some(slot) = self.slot.take() {
                self.bridge.ring.discard_write(slot);
            }
        }
    }

    impl Drop for LinuxPreviewBridge {
        fn drop(&mut self) {
            self.render_state
                .renderer
                .write()
                .free_texture(&self.texture_id);
        }
    }

    impl LinuxPreviewBridge {
        pub fn new(
            render_state: &egui_wgpu::RenderState,
            geometry: PreviewGeometry,
        ) -> anyhow::Result<Arc<Self>> {
            let hal_device = unsafe {
                render_state
                    .device
                    .as_hal::<wgpu_hal::api::Vulkan>()
                    .ok_or_else(|| anyhow!("preview requires the Vulkan wgpu backend"))?
            };
            let enabled = hal_device.enabled_device_extensions();
            anyhow::ensure!(
                enabled.contains(&ash::khr::external_memory_fd::NAME),
                "wgpu Vulkan device did not enable VK_KHR_external_memory_fd"
            );
            let raw_device = hal_device.raw_device();
            let memory_properties = unsafe {
                hal_device
                    .shared_instance()
                    .raw_instance()
                    .get_physical_device_memory_properties(hal_device.raw_physical_device())
            };
            let external_memory = ash::khr::external_memory_fd::Device::new(
                hal_device.shared_instance().raw_instance(),
                raw_device,
            );

            let mut slots = Vec::with_capacity(SLOT_COUNT);
            for index in 0..SLOT_COUNT {
                let mut external_info = vk::ExternalMemoryBufferCreateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
                let buffer_info = vk::BufferCreateInfo::default()
                    .size(geometry.buffer_size())
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut external_info);
                let raw_buffer = unsafe { raw_device.create_buffer(&buffer_info, None) }
                    .context("create exportable Vulkan preview buffer")?;
                let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw_buffer) };
                let memory_type_index = (0..memory_properties.memory_type_count)
                    .find(|index| {
                        requirements.memory_type_bits & (1 << index) != 0
                            && memory_properties.memory_types[*index as usize]
                                .property_flags
                                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    })
                    .ok_or_else(|| anyhow!("no device-local Vulkan memory for preview buffer"))?;
                let mut export_info = vk::ExportMemoryAllocateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
                let mut dedicated_info =
                    vk::MemoryDedicatedAllocateInfo::default().buffer(raw_buffer);
                let allocation_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type_index)
                    .push_next(&mut export_info)
                    .push_next(&mut dedicated_info);
                let raw_memory = unsafe { raw_device.allocate_memory(&allocation_info, None) }
                    .context("allocate exportable Vulkan preview memory")?;
                if let Err(error) =
                    unsafe { raw_device.bind_buffer_memory(raw_buffer, raw_memory, 0) }
                {
                    unsafe {
                        raw_device.free_memory(raw_memory, None);
                        raw_device.destroy_buffer(raw_buffer, None);
                    }
                    return Err(error).context("bind Vulkan preview memory");
                }
                let fd_info = vk::MemoryGetFdInfoKHR::default()
                    .memory(raw_memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
                let fd = unsafe { external_memory.get_memory_fd(&fd_info) }
                    .context("export Vulkan preview memory fd")?;
                let import_file = unsafe { File::from_raw_fd(fd) };
                let hal_buffer = unsafe {
                    wgpu_hal::vulkan::Buffer::from_raw_managed(
                        raw_buffer,
                        raw_memory,
                        0,
                        requirements.size,
                    )
                };
                let buffer = unsafe {
                    render_state
                        .device
                        .create_buffer_from_hal::<wgpu_hal::api::Vulkan>(
                            hal_buffer,
                            &wgpu::BufferDescriptor {
                                label: Some(&format!("cuda-preview-slot-{index}")),
                                size: geometry.buffer_size(),
                                usage: wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            },
                        )
                };
                slots.push(PreviewSlot {
                    buffer,
                    import_file: Mutex::new(Some(import_file)),
                    cuda: Mutex::new(None),
                });
            }
            drop(hal_device);

            let texture = render_state
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("cuda-preview-texture"),
                    size: wgpu::Extent3d {
                        width: geometry.width(),
                        height: geometry.height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let texture_id = render_state.renderer.write().register_native_texture(
                &render_state.device,
                &view,
                wgpu::FilterMode::Linear,
            );
            Ok(Arc::new(Self {
                geometry,
                ring: Arc::new(PreviewRingState::new(SLOT_COUNT)),
                slots: slots.into_boxed_slice(),
                render_state: render_state.clone(),
                texture,
                texture_id,
            }))
        }

        pub fn publish(
            &self,
            gpu: &crate::backend::ComputeOps,
            chw: &Buffer<f32>,
        ) -> anyhow::Result<bool> {
            let Some(write) = self.stage(gpu, chw)? else {
                return Ok(false);
            };
            gpu.sync()?;
            write.commit();
            Ok(true)
        }

        /// Enqueue CUDA conversion without introducing a stream boundary.
        /// Call `commit` only after a later synchronization has completed.
        pub fn stage<'a>(
            &'a self,
            gpu: &crate::backend::ComputeOps,
            chw: &Buffer<f32>,
        ) -> anyhow::Result<Option<LinuxPreviewWrite<'a>>> {
            let Some(index) = self.ring.acquire() else {
                return Ok(None);
            };
            let result = (|| {
                let slot = &self.slots[index];
                let mut cuda = slot.cuda.lock().expect("preview CUDA slot mutex poisoned");
                if cuda.is_none() {
                    let file = slot
                        .import_file
                        .lock()
                        .expect("preview import fd mutex poisoned")
                        .take()
                        .ok_or_else(|| anyhow!("preview memory fd was already consumed"))?;
                    gpu.stream.context().bind_to_thread()?;
                    use std::os::fd::AsRawFd;
                    let external_memory = unsafe {
                        result::external_memory::import_external_memory_opaque_fd(
                            file.as_raw_fd(),
                            self.geometry.buffer_size(),
                        )
                    }?;
                    let file = ManuallyDrop::new(file);
                    let _ = &file;
                    let device_ptr = unsafe {
                        result::external_memory::get_mapped_buffer(
                            external_memory,
                            0,
                            self.geometry.buffer_size(),
                        )
                    }?;
                    *cuda = Some(CudaInteropBuffer {
                        external_memory,
                        device_ptr,
                        len: self.geometry.buffer_size() as usize,
                        stream: gpu.stream.clone(),
                        completion: gpu.stream.context().new_event(None)?,
                        written: false,
                    });
                }
                gpu.chw_f32_to_rgba_u8_pitched(
                    chw,
                    cuda.as_mut().expect("CUDA preview slot initialized"),
                    self.geometry.height(),
                    self.geometry.width(),
                    self.geometry.row_bytes(),
                )?;
                Ok::<_, anyhow::Error>(())
            })();
            match result {
                Ok(()) => Ok(Some(LinuxPreviewWrite {
                    bridge: self,
                    slot: Some(index),
                })),
                Err(error) => {
                    self.ring.discard_write(index);
                    Err(error)
                }
            }
        }

        /// Copy the newest CUDA buffer into the texture sampled by egui.
        pub fn consume_latest(&self) -> bool {
            let Some(index) = self.ring.take_latest() else {
                return false;
            };
            let mut encoder =
                self.render_state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("cuda-preview-copy"),
                    });
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.slots[index].buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.geometry.row_bytes()),
                        rows_per_image: Some(self.geometry.height()),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.geometry.width(),
                    height: self.geometry.height(),
                    depth_or_array_layers: 1,
                },
            );
            self.render_state.queue.submit([encoder.finish()]);
            let ring = Arc::clone(&self.ring);
            self.render_state
                .queue
                .on_submitted_work_done(move || ring.release(index));
            true
        }

        pub const fn texture_id(&self) -> egui::TextureId {
            self.texture_id
        }

        pub fn size(&self) -> egui::Vec2 {
            egui::vec2(self.geometry.width() as f32, self.geometry.height() as f32)
        }

        pub const fn geometry(&self) -> PreviewGeometry {
            self.geometry
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::LinuxPreviewBridge;

impl PreviewGeometry {
    pub fn new(width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let packed = width.checked_mul(4)?;
        let row_bytes = packed.checked_add(WGPU_COPY_ALIGNMENT - 1)? / WGPU_COPY_ALIGNMENT
            * WGPU_COPY_ALIGNMENT;
        let buffer_size = u64::from(row_bytes).checked_mul(u64::from(height))?;
        Some(Self {
            width,
            height,
            row_bytes,
            buffer_size,
        })
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn height(self) -> u32 {
        self.height
    }

    pub const fn row_bytes(self) -> u32 {
        self.row_bytes
    }

    pub const fn buffer_size(self) -> u64 {
        self.buffer_size
    }
}

/// Lock-free latest-frame ring.
///
/// `FREE -> WRITING -> READY -> READING -> FREE`. A producer can never acquire
/// a slot retained by Vulkan, and the consumer atomically discards older READY
/// frames when a newer one is available.
#[derive(Debug)]
pub struct PreviewRingState {
    states: Box<[AtomicU8]>,
    sequences: Box<[AtomicU64]>,
    next_sequence: AtomicU64,
}

impl PreviewRingState {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "preview ring must contain at least one slot");
        Self {
            states: (0..capacity).map(|_| AtomicU8::new(FREE)).collect(),
            sequences: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            next_sequence: AtomicU64::new(1),
        }
    }

    pub fn acquire(&self) -> Option<usize> {
        self.states.iter().enumerate().find_map(|(index, state)| {
            state
                .compare_exchange(FREE, WRITING, Ordering::AcqRel, Ordering::Relaxed)
                .ok()
                .map(|_| index)
        })
    }

    pub fn publish(&self, slot: usize) {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        self.sequences[slot].store(sequence, Ordering::Relaxed);
        self.states[slot]
            .compare_exchange(WRITING, READY, Ordering::Release, Ordering::Relaxed)
            .expect("only an acquired preview slot can be published");
    }

    pub fn discard_write(&self, slot: usize) {
        self.states[slot]
            .compare_exchange(WRITING, FREE, Ordering::Release, Ordering::Relaxed)
            .expect("only an acquired preview slot can be discarded");
    }

    pub fn take_latest(&self) -> Option<usize> {
        for _ in 0..self.states.len() {
            let latest = self
                .states
                .iter()
                .enumerate()
                .filter(|(_, state)| state.load(Ordering::Acquire) == READY)
                .max_by_key(|(index, _)| self.sequences[*index].load(Ordering::Relaxed))
                .map(|(index, _)| index)?;
            let latest_sequence = self.sequences[latest].load(Ordering::Relaxed);
            if self.states[latest]
                .compare_exchange(READY, READING, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            for (index, state) in self.states.iter().enumerate() {
                if index != latest
                    && self.sequences[index].load(Ordering::Relaxed) <= latest_sequence
                {
                    let _ =
                        state.compare_exchange(READY, FREE, Ordering::AcqRel, Ordering::Relaxed);
                }
            }
            return Some(latest);
        }
        None
    }

    pub fn release(&self, slot: usize) {
        self.states[slot]
            .compare_exchange(READING, FREE, Ordering::Release, Ordering::Relaxed)
            .expect("only a consumed preview slot can be released");
    }

    pub fn is_free(&self, slot: usize) -> bool {
        self.states[slot].load(Ordering::Acquire) == FREE
    }
}
