//! Shared low-level IoBinding helpers for GPU-native ONNX inference.
//!
//! All hot-path inference calls (detect, recognize, swap) use this module
//! to bind CUDA device pointers directly into ort sessions, avoiding the
//! GPU → CPU → GPU roundtrip that `Tensor::from_array` incurs.

use std::ffi::CString;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ort::AsPointer;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};

use crate::models::manager::{BindingFencePolicy, ModelManager};

/// RAII wrapper around an OrtValue — releases the underlying tensor on drop.
pub struct OrtValueGuard(pub(crate) *mut ort_sys::OrtValue);

impl Drop for OrtValueGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                (ort::api().ReleaseValue)(self.0);
            }
        }
    }
}

/// Create an OrtValue tensor view over a raw CUDA device pointer (zero-copy).
/// Equivalent of Python `bind_input(buffer_ptr=tensor.data_ptr())`.
///
/// # Safety
/// Caller must ensure:
/// - `device_ptr` points to at least `product(shape) * sizeof(f32)` bytes
/// - The memory remains valid for the lifetime of the returned guard
/// - The memory is accessible on the CUDA device matched by `mem_info`
pub unsafe fn create_cuda_tensor_f32(
    mem_info: &MemoryInfo,
    device_ptr: u64,
    shape: &[i64],
) -> anyhow::Result<OrtValueGuard> {
    let mut elements: usize = 1;
    for &d in shape {
        elements *= d as usize;
    }
    let bytes = elements * core::mem::size_of::<f32>();

    let mut value: *mut ort_sys::OrtValue = core::ptr::null_mut();
    let status = unsafe {
        (ort::api().CreateTensorWithDataAsOrtValue)(
            mem_info.ptr(),
            device_ptr as *mut core::ffi::c_void,
            bytes,
            shape.as_ptr(),
            shape.len(),
            ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut value,
        )
    };
    unsafe { ort::error::Error::result_from_status(status) }?;
    anyhow::ensure!(
        !value.is_null(),
        "CreateTensorWithDataAsOrtValue returned null OrtValue"
    );
    Ok(OrtValueGuard(value))
}

/// Bind an OrtValue to a session input via low-level C API.
///
/// # Safety
/// `binding` must be a valid IoBinding not currently being used by another thread.
unsafe fn bind_input_raw(
    binding: &mut IoBinding,
    name: &str,
    value: &OrtValueGuard,
) -> anyhow::Result<()> {
    let cname = CString::new(name)?;
    let status =
        unsafe { (ort::api().BindInput)(binding.ptr().cast_mut(), cname.as_ptr(), value.0) };
    unsafe { ort::error::Error::result_from_status(status) }?;
    Ok(())
}

/// Bind an OrtValue to a session output via low-level C API.
///
/// # Safety
/// Same constraints as `bind_input_raw`. Output buffer writes directly into
/// the user-provided device buffer (zero-copy).
unsafe fn bind_output_raw(
    binding: &mut IoBinding,
    name: &str,
    value: &OrtValueGuard,
) -> anyhow::Result<()> {
    let cname = CString::new(name)?;
    let status =
        unsafe { (ort::api().BindOutput)(binding.ptr().cast_mut(), cname.as_ptr(), value.0) };
    unsafe { ort::error::Error::result_from_status(status) }?;
    Ok(())
}

trait BindingProtocol {
    fn bind_all(&mut self) -> anyhow::Result<()>;
    fn synchronize_inputs(&mut self) -> anyhow::Result<()>;
    fn run(&mut self) -> anyhow::Result<()>;
    fn synchronize_outputs(&mut self) -> anyhow::Result<()>;
    fn clear(&mut self);
}

struct ClearBindingOnDrop<'a, P: BindingProtocol + ?Sized> {
    protocol: &'a mut P,
}

impl<P: BindingProtocol + ?Sized> Drop for ClearBindingOnDrop<'_, P> {
    fn drop(&mut self) {
        self.protocol.clear();
    }
}

fn execute_binding_protocol<P: BindingProtocol + ?Sized>(
    policy: BindingFencePolicy,
    protocol: &mut P,
) -> anyhow::Result<()> {
    let guard = ClearBindingOnDrop { protocol };
    guard.protocol.bind_all()?;
    if policy == BindingFencePolicy::FenceInputsAndOutputs {
        guard.protocol.synchronize_inputs()?;
    }
    guard.protocol.run()?;
    if policy == BindingFencePolicy::FenceInputsAndOutputs {
        guard.protocol.synchronize_outputs()?;
    }
    Ok(())
}

struct OrtBindingProtocol<'session, 'binding, 'values> {
    session: &'session mut Session,
    binding: &'binding mut IoBinding,
    inputs: &'values [(&'values str, &'values OrtValueGuard)],
    outputs: &'values [(&'values str, &'values OrtValueGuard)],
}

impl BindingProtocol for OrtBindingProtocol<'_, '_, '_> {
    fn bind_all(&mut self) -> anyhow::Result<()> {
        for (name, value) in self.inputs {
            unsafe { bind_input_raw(self.binding, name, value)? };
        }
        for (name, value) in self.outputs {
            unsafe { bind_output_raw(self.binding, name, value)? };
        }
        Ok(())
    }

    fn synchronize_inputs(&mut self) -> anyhow::Result<()> {
        self.binding.synchronize_inputs()?;
        Ok(())
    }

    fn run(&mut self) -> anyhow::Result<()> {
        let outputs = self.session.run_binding(self.binding)?;
        drop(outputs);
        Ok(())
    }

    fn synchronize_outputs(&mut self) -> anyhow::Result<()> {
        self.binding.synchronize_outputs()?;
        Ok(())
    }

    fn clear(&mut self) {
        self.binding.clear();
    }
}

/// Bind caller-owned CUDA tensors, run one cached IoBinding, and clear it on
/// every exit path. Device-wide fences are used only when stream ordering is
/// not sufficient for the selected provider.
pub fn run_bound_values(
    manager: &mut ModelManager,
    stream: &CudaStream,
    session_name: &str,
    inputs: &[(&str, &OrtValueGuard)],
    outputs: &[(&str, &OrtValueGuard)],
) -> anyhow::Result<()> {
    let policy = manager.binding_fence_policy(stream.cu_stream() as *mut ());
    let (session, binding) = manager.session_and_binding(session_name)?;
    let mut protocol = OrtBindingProtocol {
        session,
        binding,
        inputs,
        outputs,
    };
    execute_binding_protocol(policy, &mut protocol)
}

/// Run a one-input/one-output model directly against caller-owned CUDA buffers.
/// The cached IoBinding avoids an ORT output allocation and the following D2D copy.
#[allow(clippy::too_many_arguments)]
pub fn run_bound_f32(
    manager: &mut ModelManager,
    stream: &CudaStream,
    session_name: &str,
    input_name: &str,
    input: &CudaSlice<f32>,
    input_shape: &[i64],
    output_name: &str,
    output: &mut CudaSlice<f32>,
    output_shape: &[i64],
) -> anyhow::Result<()> {
    let memory = MemoryInfo::new(
        AllocationDevice::CUDA,
        manager.device_id(),
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let (input_ptr, _input_guard) = input.device_ptr(stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(stream);
    let input_value = unsafe { create_cuda_tensor_f32(&memory, input_ptr, input_shape)? };
    let output_value = unsafe { create_cuda_tensor_f32(&memory, output_ptr, output_shape)? };
    run_bound_values(
        manager,
        stream,
        session_name,
        &[(input_name, &input_value)],
        &[(output_name, &output_value)],
    )
}

#[cfg(test)]
mod tests {
    use super::{BindingProtocol, execute_binding_protocol};
    use crate::models::manager::BindingFencePolicy;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Failure {
        Bind,
        InputFence,
        Run,
        OutputFence,
    }

    #[derive(Default)]
    struct FakeProtocol {
        actions: Vec<&'static str>,
        failure: Option<Failure>,
    }

    impl FakeProtocol {
        fn action(&mut self, name: &'static str, failure: Failure) -> anyhow::Result<()> {
            self.actions.push(name);
            if self.failure == Some(failure) {
                anyhow::bail!("injected {name} failure");
            }
            Ok(())
        }
    }

    impl BindingProtocol for FakeProtocol {
        fn bind_all(&mut self) -> anyhow::Result<()> {
            self.action("bind", Failure::Bind)
        }

        fn synchronize_inputs(&mut self) -> anyhow::Result<()> {
            self.action("sync_inputs", Failure::InputFence)
        }

        fn run(&mut self) -> anyhow::Result<()> {
            self.action("run", Failure::Run)
        }

        fn synchronize_outputs(&mut self) -> anyhow::Result<()> {
            self.action("sync_outputs", Failure::OutputFence)
        }

        fn clear(&mut self) {
            self.actions.push("clear");
        }
    }

    #[test]
    fn same_stream_protocol_runs_without_device_fences_and_clears() {
        let mut protocol = FakeProtocol::default();
        execute_binding_protocol(BindingFencePolicy::SameCudaStream, &mut protocol).unwrap();
        assert_eq!(protocol.actions, ["bind", "run", "clear"]);
    }

    #[test]
    fn fenced_protocol_orders_both_fences_and_clears() {
        let mut protocol = FakeProtocol::default();
        execute_binding_protocol(BindingFencePolicy::FenceInputsAndOutputs, &mut protocol).unwrap();
        assert_eq!(
            protocol.actions,
            ["bind", "sync_inputs", "run", "sync_outputs", "clear"]
        );
    }

    #[test]
    fn every_injected_error_still_clears_once() {
        for failure in [
            Failure::Bind,
            Failure::InputFence,
            Failure::Run,
            Failure::OutputFence,
        ] {
            let mut protocol = FakeProtocol {
                actions: Vec::new(),
                failure: Some(failure),
            };
            assert!(
                execute_binding_protocol(BindingFencePolicy::FenceInputsAndOutputs, &mut protocol,)
                    .is_err()
            );
            assert_eq!(protocol.actions.last(), Some(&"clear"));
            assert_eq!(
                protocol
                    .actions
                    .iter()
                    .filter(|action| **action == "clear")
                    .count(),
                1
            );
        }
    }

    #[test]
    fn panic_unwinding_still_clears() {
        struct PanicProtocol<'a>(&'a mut bool);

        impl BindingProtocol for PanicProtocol<'_> {
            fn bind_all(&mut self) -> anyhow::Result<()> {
                panic!("injected panic")
            }
            fn synchronize_inputs(&mut self) -> anyhow::Result<()> {
                unreachable!()
            }
            fn run(&mut self) -> anyhow::Result<()> {
                unreachable!()
            }
            fn synchronize_outputs(&mut self) -> anyhow::Result<()> {
                unreachable!()
            }
            fn clear(&mut self) {
                *self.0 = true;
            }
        }

        let mut cleared = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut protocol = PanicProtocol(&mut cleared);
            let _ =
                execute_binding_protocol(BindingFencePolicy::FenceInputsAndOutputs, &mut protocol);
        }));
        assert!(result.is_err());
        assert!(cleared);
    }
}
