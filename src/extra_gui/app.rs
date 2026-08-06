use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use super::editor::{MediaKind, MediaRole};
use super::faces::{ARC_FACE_MODEL, EmbeddingMergeMethod, FaceCrop};
use super::panels;
use super::runtime::{
    AnalyzeRequest, EditorJobPhase, EditorJobState, EditorRuntimeEvent, EditorRuntimeHandle,
};
use super::state::{ExtraGuiPanel, ExtraGuiState};
use super::workspace::WorkspaceDocument;
use super::{ControlValue, EditorRuntimeConfig};

pub struct ExtraGuiApp {
    models_dir: PathBuf,
    state: ExtraGuiState,
    last_autosave: Instant,
    runtime: EditorRuntimeHandle,
    job: EditorJobState,
}

impl ExtraGuiApp {
    pub fn new(models_dir: PathBuf) -> Self {
        Self::new_with_render_state(models_dir, None)
    }

    fn new_with_render_state(
        models_dir: PathBuf,
        render_state: Option<egui_wgpu::RenderState>,
    ) -> Self {
        let state = ExtraGuiState::new(&models_dir);
        let runtime =
            EditorRuntimeHandle::spawn_with_render_state(models_dir.clone(), render_state)
                .expect("extra GUI worker thread must be available");
        Self {
            models_dir,
            state,
            last_autosave: Instant::now(),
            runtime,
            job: EditorJobState::default(),
        }
    }

    pub const fn active_panel(&self) -> ExtraGuiPanel {
        self.state.active_panel
    }

    pub fn select_panel(&mut self, panel: ExtraGuiPanel) {
        self.state.active_panel = panel;
    }

    fn install_style(ctx: &egui::Context) {
        crate::app::install_product_theme(ctx);
    }

    fn sync_preview_texture(&mut self, ctx: &egui::Context) {
        let selected = self.state.media.selected(MediaRole::Target);
        if selected == self.state.preview_loaded {
            return;
        }
        self.state.preview_loaded = selected;
        self.state.source_texture = None;
        self.state.source_gpu_texture = None;
        self.state.output_texture = None;
        self.state.output_gpu_texture = None;
        self.state.latest_output = None;
        let Some(item) = selected.and_then(|id| self.state.media.item(id)) else {
            return;
        };
        if item.kind != MediaKind::Image {
            self.state.status = "Video target ready for timeline decoding".to_owned();
            return;
        }
        match image::open(&item.path) {
            Ok(image) => {
                let rgba = image.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                self.state.source_texture = Some(ctx.load_texture(
                    format!("extra-target-{}", item.path.display()),
                    color,
                    egui::TextureOptions::LINEAR,
                ));
                self.state.status = format!("Loaded {}", item.path.display());
                self.state.preview.reset();
            }
            Err(error) => {
                self.state.status = format!("Could not open {}: {error}", item.path.display());
            }
        }
    }

    fn sync_face_textures(&mut self, ctx: &egui::Context) {
        let live_ids = self
            .state
            .faces
            .targets()
            .map(|face| face.id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        self.state
            .face_textures
            .retain(|id, _| live_ids.contains(id));
        for face in self.state.faces.targets() {
            if self.state.face_textures.contains_key(&face.id) {
                continue;
            }
            let Some(crop) = &face.crop else {
                continue;
            };
            let image =
                egui::ColorImage::from_rgb([crop.width as usize, crop.height as usize], &crop.rgb);
            self.state.face_textures.insert(
                face.id.clone(),
                ctx.load_texture(
                    format!("target-face-{}", face.id),
                    image,
                    egui::TextureOptions::LINEAR,
                ),
            );
        }

        let live_source_ids = self
            .state
            .faces
            .sources()
            .map(|source| source.id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        self.state
            .source_face_textures
            .retain(|id, _| live_source_ids.contains(id));
        for source in self.state.faces.sources() {
            if self.state.source_face_textures.contains_key(&source.id) {
                continue;
            }
            let Some(path) = source
                .media_id
                .and_then(|media_id| self.state.media.item(media_id))
                .map(|media| &media.path)
            else {
                continue;
            };
            let Ok(image) = image::open(path) else {
                continue;
            };
            let thumbnail = image.thumbnail(128, 128).to_rgb8();
            let color = egui::ColorImage::from_rgb(
                [thumbnail.width() as usize, thumbnail.height() as usize],
                thumbnail.as_raw(),
            );
            self.state.source_face_textures.insert(
                source.id.clone(),
                ctx.load_texture(
                    format!("source-face-{}", source.id),
                    color,
                    egui::TextureOptions::LINEAR,
                ),
            );
        }
    }

    fn document(&self) -> WorkspaceDocument {
        let mut document =
            WorkspaceDocument::new(self.state.workspace_name.clone(), &self.state.catalog)
                .expect("editor state originates from the embedded catalog");
        document.media = self.state.media.clone();
        document.faces = self.state.faces.clone();
        document.controls = self.state.controls.clone();
        document.preview = self.state.preview.clone();
        document.timeline = self.state.timeline.clone();
        document.output_directory = self.state.output_directory.clone();
        document
    }

    fn apply_document(&mut self, path: Option<PathBuf>, document: WorkspaceDocument) {
        self.state.workspace_name = document.name;
        self.state.workspace_path = path;
        self.state.media = document.media;
        self.state.faces = document.faces;
        self.state.controls = document.controls;
        self.state.preview = document.preview;
        self.state.timeline = document.timeline;
        self.state.output_directory = document.output_directory;
        self.state.preview_loaded = None;
        self.state.source_texture = None;
        self.state.source_gpu_texture = None;
        self.state.output_texture = None;
        self.state.output_gpu_texture = None;
        self.state.latest_output = None;
        self.state.face_textures.clear();
        self.state.source_face_textures.clear();
        self.state.dirty = false;
        self.state.status = "Workspace loaded".to_owned();
    }

    fn save_native(&mut self, force_dialog: bool) {
        let path = if !force_dialog {
            self.state.workspace_path.clone()
        } else {
            None
        }
        .or_else(|| {
            rfd::FileDialog::new()
                .add_filter("noperson workspace", &["json"])
                .set_file_name("workspace.noperson.json")
                .save_file()
        });
        let Some(path) = path else {
            return;
        };
        match self.document().save_atomic(&path) {
            Ok(()) => {
                self.state.workspace_name = path
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("workspace")
                    .to_owned();
                self.state.workspace_path = Some(path.clone());
                self.state.dirty = false;
                self.state.status = format!("Saved {}", path.display());
                self.last_autosave = Instant::now();
            }
            Err(error) => self.state.status = error.to_string(),
        }
    }

    fn open_workspace(&mut self, crossswap: bool) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON workspace", &["json"])
            .pick_file()
        else {
            return;
        };
        let result = if crossswap {
            std::fs::read(&path)
                .map_err(super::workspace::WorkspaceError::from)
                .and_then(|bytes| serde_json::from_slice(&bytes).map_err(Into::into))
                .and_then(|value| {
                    WorkspaceDocument::from_crossswap_value(
                        path.file_stem()
                            .and_then(|name| name.to_str())
                            .unwrap_or("CrossSwap workspace"),
                        value,
                        &self.state.catalog,
                    )
                })
        } else {
            WorkspaceDocument::load(&path, &self.state.catalog)
        };
        match result {
            Ok(document) => self.apply_document(Some(path), document),
            Err(error) => self.state.status = error.to_string(),
        }
    }

    fn export_crossswap(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("CrossSwap workspace", &["json"])
            .set_file_name("crosswap-workspace.json")
            .save_file()
        else {
            return;
        };
        let result = self
            .document()
            .to_crossswap_value(&self.state.catalog)
            .and_then(|value| {
                let bytes = serde_json::to_vec_pretty(&value)?;
                std::fs::write(&path, bytes).map_err(Into::into)
            });
        match result {
            Ok(()) => self.state.status = format!("Exported {}", path.display()),
            Err(error) => self.state.status = error.to_string(),
        }
    }

    fn autosave(&mut self) {
        if !self.state.dirty || self.last_autosave.elapsed() < Duration::from_secs(30) {
            return;
        }
        let Some(path) = self.state.workspace_path.as_ref() else {
            return;
        };
        let autosave = PathBuf::from(format!("{}.autosave", path.display()));
        match self.document().save_atomic(&autosave) {
            Ok(()) => self.last_autosave = Instant::now(),
            Err(error) => self.state.status = format!("Autosave failed: {error}"),
        }
    }

    fn merge_method(controls: &super::ControlState) -> EmbeddingMergeMethod {
        match controls.get("EmbMergeMethodSelection") {
            Some(ControlValue::Choice(value)) if value == "Median" => EmbeddingMergeMethod::Median,
            _ => EmbeddingMergeMethod::Mean,
        }
    }

    fn selected_target(&self) -> Option<super::editor::MediaItem> {
        self.state
            .media
            .selected(MediaRole::Target)
            .and_then(|id| self.state.media.item(id))
            .cloned()
    }

    fn begin_analysis(&mut self) {
        if let Err(error) = self.job.begin(EditorJobPhase::Analyzing) {
            self.state.status = error.to_string();
            return;
        }
        let Some(target) = self.selected_target() else {
            self.job.settle();
            self.state.status = "Select target media first".to_owned();
            return;
        };
        let runtime = match EditorRuntimeConfig::from_controls(&self.state.controls) {
            Ok(runtime) => runtime,
            Err(error) => {
                self.job.fail();
                self.state.status = error.to_string();
                return;
            }
        };
        let source_paths = self
            .state
            .faces
            .sources()
            .filter_map(|source| {
                let media = source.media_id.and_then(|id| self.state.media.item(id))?;
                Some((source.id.clone(), media.path.clone()))
            })
            .collect();
        if let Err(error) = self.runtime.analyze(AnalyzeRequest {
            target_media: target.id,
            target_path: target.path,
            target_kind: target.kind,
            frame_index: self.state.timeline.current_frame(),
            source_paths,
            runtime,
        }) {
            self.job.fail();
            self.state.status = error.to_string();
        } else {
            self.state.status = "Analyzing target and source identities".to_owned();
        }
    }

    fn compile_request(
        &self,
        controls: &super::ControlState,
        faces: &super::FaceWorkspace,
    ) -> Result<super::EditorEngineRequest, String> {
        EditorRuntimeConfig::from_controls(controls)
            .and_then(|runtime| {
                runtime.compile_engine_request(
                    faces,
                    Self::merge_method(controls),
                    &self.models_dir,
                    0,
                )
            })
            .map_err(|error| error.to_string())
    }

    fn compile_marker_requests(&self) -> Result<BTreeMap<u64, super::EditorEngineRequest>, String> {
        self.state
            .timeline
            .markers()
            .map(|(frame, marker)| {
                let mut faces = self.state.faces.clone();
                for (face_id, controls) in &marker.face_controls {
                    if let Some(face) = faces.target_mut(face_id) {
                        face.controls = controls.clone();
                    }
                }
                self.compile_request(&marker.controls, &faces)
                    .map(|request| (frame, request))
                    .map_err(|error| format!("Marker {frame}: {error}"))
            })
            .collect()
    }

    fn begin_preview(&mut self) {
        if let Err(error) = self.job.begin(EditorJobPhase::Building) {
            self.state.status = error.to_string();
            return;
        }
        let Some(target) = self.selected_target() else {
            self.job.settle();
            self.state.status = "Select target media first".to_owned();
            return;
        };
        let request = match self.compile_request(&self.state.controls, &self.state.faces) {
            Ok(request) => request,
            Err(error) => {
                self.job.fail();
                self.state.status = error;
                return;
            }
        };
        if let Err(error) = self.runtime.preview(
            target.path,
            target.kind,
            self.state.timeline.current_frame(),
            request,
        ) {
            self.job.fail();
            self.state.status = error.to_string();
        } else {
            self.state.status = "Building atomic preview generation".to_owned();
        }
    }

    fn begin_playback(&mut self) {
        if let Err(error) = self.job.begin(EditorJobPhase::Building) {
            self.state.timeline.set_playing(false);
            self.state.status = error.to_string();
            return;
        }
        let Some(target) = self.selected_target() else {
            self.job.settle();
            self.state.timeline.set_playing(false);
            self.state.status = "Select target video first".to_owned();
            return;
        };
        if target.kind != MediaKind::Video {
            self.job.settle();
            self.state.timeline.set_playing(false);
            self.state.status = "Playback requires a video target".to_owned();
            return;
        }
        let runtime = match EditorRuntimeConfig::from_controls(&self.state.controls) {
            Ok(runtime) => runtime,
            Err(error) => {
                self.job.fail();
                self.state.timeline.set_playing(false);
                self.state.status = error.to_string();
                return;
            }
        };
        let request = match runtime.compile_engine_request(
            &self.state.faces,
            Self::merge_method(&self.state.controls),
            &self.models_dir,
            0,
        ) {
            Ok(request) => request,
            Err(error) => {
                self.job.fail();
                self.state.timeline.set_playing(false);
                self.state.status = error.to_string();
                return;
            }
        };
        let markers = match self.compile_marker_requests() {
            Ok(markers) => markers,
            Err(error) => {
                self.job.fail();
                self.state.timeline.set_playing(false);
                self.state.status = error;
                return;
            }
        };
        if let Err(error) = self.runtime.playback(
            target.path,
            self.state.timeline.current_frame(),
            runtime.playback_fps,
            request,
            markers,
        ) {
            self.job.fail();
            self.state.timeline.set_playing(false);
            self.state.status = error.to_string();
        }
    }

    fn consume_timeline_intents(&mut self) {
        if std::mem::take(&mut self.state.pause_requested) {
            self.cancel_job();
        }
        if std::mem::take(&mut self.state.play_requested) {
            self.begin_playback();
        }
        if std::mem::take(&mut self.state.record_requested) {
            self.begin_record();
        }
    }

    fn consume_analysis_intent(&mut self) {
        if !self.state.analysis_requested {
            return;
        }
        if !matches!(
            self.job.phase(),
            EditorJobPhase::Idle | EditorJobPhase::Failed
        ) {
            return;
        }
        self.state.analysis_requested = false;
        self.begin_analysis();
    }

    fn apply_exact_marker(&mut self) {
        let Some(marker) = self
            .state
            .timeline
            .marker_at(self.state.timeline.current_frame())
            .cloned()
        else {
            return;
        };
        self.state.controls = marker.controls;
        for (face_id, controls) in marker.face_controls {
            if let Some(face) = self.state.faces.target_mut(&face_id) {
                face.controls = controls;
            }
        }
    }

    fn keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let keys = ctx.input(|input| {
            (
                input.key_pressed(egui::Key::F11),
                input.key_pressed(egui::Key::Space),
                input.key_pressed(egui::Key::C),
                input.key_pressed(egui::Key::V),
                input.key_pressed(egui::Key::A),
                input.key_pressed(egui::Key::D),
                input.key_pressed(egui::Key::Z),
                input.key_pressed(egui::Key::Q),
                input.key_pressed(egui::Key::W),
                input.key_pressed(egui::Key::F),
                input.modifiers.alt,
                input.key_pressed(egui::Key::P),
                input.key_pressed(egui::Key::S),
                input.key_pressed(egui::Key::R),
            )
        });
        if keys.0 {
            self.state.fullscreen = !self.state.fullscreen;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.state.fullscreen));
        }
        if keys.1 {
            self.state.timeline.toggle_playback();
            if self.state.timeline.is_playing() {
                self.state.play_requested = true;
            } else {
                self.state.pause_requested = true;
            }
        }
        let delta = if keys.2 {
            -1
        } else if keys.3 {
            1
        } else if keys.4 {
            -30
        } else if keys.5 {
            30
        } else {
            0
        };
        if delta != 0 {
            self.state.timeline.step(delta);
            self.apply_exact_marker();
        }
        if keys.6 {
            self.state.timeline.seek(0);
            self.apply_exact_marker();
        }
        if keys.7
            && let Some(frame) = self.state.timeline.previous_marker()
        {
            self.state.timeline.seek(frame);
            self.apply_exact_marker();
        }
        if keys.8
            && let Some(frame) = self.state.timeline.next_marker()
        {
            self.state.timeline.seek(frame);
            self.apply_exact_marker();
        }
        if keys.9 {
            let frame = self.state.timeline.current_frame();
            if keys.10 {
                self.state.timeline.remove_marker(frame);
            } else {
                let face_controls = self
                    .state
                    .faces
                    .targets()
                    .map(|face| (face.id.clone(), face.controls.clone()))
                    .collect();
                self.state
                    .timeline
                    .add_marker_snapshot(self.state.controls.clone(), face_controls);
            }
            self.state.dirty = true;
        }
        if keys.11 {
            self.state.preview.detached = !self.state.preview.detached;
        }
        if keys.12 {
            self.begin_preview();
        }
        if keys.13 {
            self.begin_record();
        }
    }

    fn consume_dropped_media(&mut self, ctx: &egui::Context) {
        let (dropped, source_role) = ctx.input(|input| {
            (
                input
                    .raw
                    .dropped_files
                    .iter()
                    .filter_map(|file| file.path.clone())
                    .collect::<Vec<_>>(),
                input.modifiers.shift,
            )
        });
        if dropped.is_empty() {
            return;
        }
        let role = if source_role {
            MediaRole::Source
        } else {
            MediaRole::Target
        };
        let recursive = match role {
            MediaRole::Target => "TargetMediaFolderRecursiveToggle",
            MediaRole::Source => "InputFacesFolderRecursiveToggle",
        };
        let recursive = matches!(
            self.state.controls.get(recursive),
            Some(ControlValue::Toggle(true))
        );
        let mut files = Vec::new();
        for path in dropped {
            if path.is_dir() {
                match super::discover_media(&path, role, recursive) {
                    Ok(mut discovered) => files.append(&mut discovered),
                    Err(error) => self.state.status = error.to_string(),
                }
            } else {
                files.push(path);
            }
        }
        panels::add_media_paths(&mut self.state, role, files);
    }

    fn begin_record(&mut self) {
        if let Err(error) = self.job.begin(EditorJobPhase::Building) {
            self.state.status = error.to_string();
            return;
        }
        let Some(target) = self.selected_target() else {
            self.job.settle();
            self.state.status = "Select target video first".to_owned();
            return;
        };
        if target.kind != MediaKind::Video {
            self.job.settle();
            self.state.status = "Recording requires a video target".to_owned();
            return;
        }
        let mut output_dialog = rfd::FileDialog::new()
            .add_filter("MP4 video", &["mp4"])
            .set_file_name("noperson-output.mp4");
        if let Some(directory) = &self.state.output_directory {
            output_dialog = output_dialog.set_directory(directory);
        }
        let Some(output) = output_dialog.save_file() else {
            self.job.settle();
            return;
        };
        self.state.output_directory = output.parent().map(PathBuf::from);
        let initial = match self.compile_request(&self.state.controls, &self.state.faces) {
            Ok(request) => request,
            Err(error) => {
                self.job.fail();
                self.state.status = error;
                return;
            }
        };
        let markers = match self.compile_marker_requests() {
            Ok(markers) => markers,
            Err(error) => {
                self.job.fail();
                self.state.status = error;
                return;
            }
        };
        if let Err(error) = self.runtime.record(target.path, output, initial, markers) {
            self.job.fail();
            self.state.status = error.to_string();
        } else {
            self.state.status = "Building recording generation".to_owned();
        }
    }

    fn cancel_job(&mut self) {
        if self.job.cancel().is_ok() {
            self.runtime.cancel();
            self.state.status = "Cancelling at the next frame boundary".to_owned();
        }
    }

    fn runtime_actions(&mut self, ui: &mut egui::Ui) {
        let idle = matches!(
            self.job.phase(),
            EditorJobPhase::Idle | EditorJobPhase::Failed
        );
        if ui
            .add_enabled(idle, egui::Button::new("Analyze faces"))
            .clicked()
        {
            self.begin_analysis();
        }
        if ui
            .add_enabled(idle, egui::Button::new("Render preview"))
            .clicked()
        {
            self.begin_preview();
        }
        if ui
            .add_enabled(idle, egui::Button::new("Record video"))
            .clicked()
        {
            self.begin_record();
        }
        if ui.add_enabled(!idle, egui::Button::new("Cancel")).clicked() {
            self.cancel_job();
        }
        if ui
            .add_enabled(idle, egui::Button::new("Release GPU"))
            .on_hover_text("Drop loaded generations and the editor CUDA context")
            .clicked()
        {
            match self.runtime.clear_cache() {
                Ok(()) => self.state.status = "Releasing editor GPU memory".to_owned(),
                Err(error) => self.state.status = error.to_string(),
            }
        }
    }

    fn poll_runtime(&mut self, ctx: &egui::Context) {
        let events = self.runtime.try_events().collect::<Vec<_>>();
        for event in events {
            match event {
                EditorRuntimeEvent::Phase(phase) => {
                    self.job.observe(phase);
                    self.state.runtime_phase = format!("{phase:?}");
                    if phase == EditorJobPhase::Idle {
                        self.state.timeline.set_playing(false);
                    }
                    if phase != EditorJobPhase::Idle {
                        self.state.status = format!("{phase:?}");
                    }
                }
                EditorRuntimeEvent::Analyzed {
                    target_media,
                    target_faces,
                    source_faces,
                    total_frames,
                    fps,
                } => {
                    let auto_swap = matches!(
                        self.state.controls.get("AutoSwapToggle"),
                        Some(ControlValue::Toggle(true))
                    );
                    let active_target = if self.state.faces.swap_all() {
                        None
                    } else {
                        self.state.faces.selected_target().map(str::to_owned)
                    };
                    let retained_sources = if self.state.pending_auto_sources.is_empty() {
                        self.state
                            .faces
                            .sources()
                            .filter(|source| {
                                self.state
                                    .faces
                                    .source_is_assigned(active_target.as_deref(), &source.id)
                            })
                            .map(|source| source.id.clone())
                            .collect::<Vec<_>>()
                    } else {
                        self.state.pending_auto_sources.iter().cloned().collect()
                    };
                    let retained_merged = if self.state.pending_auto_merged.is_empty() {
                        self.state
                            .faces
                            .merged_identities()
                            .filter(|identity| {
                                self.state
                                    .faces
                                    .merged_is_assigned(active_target.as_deref(), &identity.id)
                            })
                            .map(|identity| identity.id.clone())
                            .collect::<Vec<_>>()
                    } else {
                        self.state.pending_auto_merged.iter().cloned().collect()
                    };
                    for (id, face) in source_faces {
                        let embeddings =
                            BTreeMap::from([(ARC_FACE_MODEL.to_owned(), face.embedding)]);
                        if let Err(error) = self.state.faces.set_source_embeddings(&id, embeddings)
                        {
                            self.state.status = error.to_string();
                        }
                    }
                    self.state.faces.clear_targets();
                    for (index, face) in target_faces.into_iter().enumerate() {
                        let digest = match crate::live::embedding_blake3(&face.embedding) {
                            Ok(digest) => digest,
                            Err(error) => {
                                self.state.status = error.to_string();
                                continue;
                            }
                        };
                        let id = format!("{}-{index}", &digest[..12]);
                        let embeddings =
                            BTreeMap::from([(ARC_FACE_MODEL.to_owned(), face.embedding)]);
                        let crop = FaceCrop {
                            width: face.crop_width,
                            height: face.crop_height,
                            rgb: face.crop_rgb,
                        };
                        if let Err(error) = self.state.faces.add_target_with_crop(
                            id.clone(),
                            Some(target_media),
                            embeddings,
                            Some(crop),
                            &self.state.catalog,
                        ) {
                            self.state.status = error.to_string();
                            continue;
                        }
                        if let Some(target) = self.state.faces.target_mut(&id) {
                            target.controls = self.state.controls.clone();
                        }
                    }
                    if auto_swap {
                        let target_ids = self
                            .state
                            .faces
                            .targets()
                            .map(|target| target.id.clone())
                            .collect::<Vec<_>>();
                        for target_id in target_ids {
                            for source_id in &retained_sources {
                                if let Err(error) =
                                    self.state.faces.assign_source(&target_id, source_id, true)
                                {
                                    self.state.status = error.to_string();
                                }
                            }
                            for merged_id in &retained_merged {
                                if let Err(error) =
                                    self.state.faces.assign_merged(&target_id, merged_id, true)
                                {
                                    self.state.status = error.to_string();
                                }
                            }
                        }
                    }
                    self.state.pending_auto_sources.clear();
                    self.state.pending_auto_merged.clear();
                    self.state.timeline = super::EditorTimeline::new(total_frames.max(1), fps);
                    self.state.dirty = true;
                    self.state.status = "Identity analysis complete".to_owned();
                }
                EditorRuntimeEvent::Preview {
                    input,
                    output,
                    playback,
                } => {
                    self.state.source_gpu_texture = None;
                    self.state.output_gpu_texture = None;
                    let source_image = egui::ColorImage::from_rgb(
                        [input.width as usize, input.height as usize],
                        &input.data,
                    );
                    self.state.source_texture = Some(ctx.load_texture(
                        "extra-source-preview",
                        source_image,
                        egui::TextureOptions::LINEAR,
                    ));
                    let output_image = egui::ColorImage::from_rgb(
                        [output.width as usize, output.height as usize],
                        &output.data,
                    );
                    self.state.output_texture = Some(ctx.load_texture(
                        "extra-output-preview",
                        output_image,
                        egui::TextureOptions::LINEAR,
                    ));
                    if !playback {
                        self.state.preview.reset();
                        self.state.active_panel = ExtraGuiPanel::Preview;
                    }
                    self.state.status = format!(
                        "Preview ready · {} detected · {} swapped",
                        output.faces_detected, output.faces_swapped
                    );
                    self.state.latest_output = Some(output);
                }
                #[cfg(target_os = "linux")]
                EditorRuntimeEvent::GpuPreview {
                    input_bridge,
                    output_bridge,
                    faces_detected,
                    faces_swapped,
                    playback,
                } => {
                    if input_bridge.consume_latest() {
                        self.state.source_gpu_texture = Some(egui::load::SizedTexture::new(
                            input_bridge.texture_id(),
                            input_bridge.size(),
                        ));
                        self.state.source_texture = None;
                    }
                    if output_bridge.consume_latest() {
                        self.state.output_gpu_texture = Some(egui::load::SizedTexture::new(
                            output_bridge.texture_id(),
                            output_bridge.size(),
                        ));
                        self.state.output_texture = None;
                    }
                    if !playback {
                        self.state.preview.reset();
                        self.state.active_panel = ExtraGuiPanel::Preview;
                    }
                    self.state.status = format!(
                        "Preview ready · {faces_detected} detected · {faces_swapped} swapped"
                    );
                    self.state.latest_output = None;
                }
                EditorRuntimeEvent::Progress { processed, total } => {
                    self.state.runtime_progress = Some((processed, total));
                    self.state.timeline.seek(processed.saturating_sub(1));
                    let action = if self.job.phase() == EditorJobPhase::Recording {
                        "Recording"
                    } else {
                        "Playback"
                    };
                    self.state.status = total.map_or_else(
                        || format!("{action} {processed} frames"),
                        |total| format!("{action} {processed}/{total}"),
                    );
                }
                EditorRuntimeEvent::PredecodeProgress { decoded, total } => {
                    self.state.runtime_progress = Some((decoded, total));
                    self.state.status = total.map_or_else(
                        || format!("Pre-decoding {decoded} frames to VRAM"),
                        |total| format!("Pre-decoding {decoded}/{total} frames to VRAM"),
                    );
                }
                EditorRuntimeEvent::Completed(path) => {
                    self.state.last_output = Some(path.clone());
                    self.state.runtime_progress = None;
                    self.state.status = format!("Recorded {}", path.display());
                }
                EditorRuntimeEvent::CacheCleared => {
                    self.state.runtime_progress = None;
                    self.state.status = "Editor GPU memory released".to_owned();
                }
                EditorRuntimeEvent::Failed(error) => {
                    self.job.fail();
                    self.state.status = error;
                }
            }
        }
    }

    fn workspace_actions(&mut self, ui: &mut egui::Ui) {
        if ui.button("New").clicked() {
            self.state = ExtraGuiState::new(&self.models_dir);
        }
        if ui.button("Open").clicked() {
            self.open_workspace(false);
        }
        if ui.button("Save").clicked() {
            self.save_native(false);
        }
        if ui.button("Save as").clicked() {
            self.save_native(true);
        }
        ui.menu_button("CrossSwap", |ui| {
            if ui.button("Import workspace").clicked() {
                ui.close();
                self.open_workspace(true);
            }
            if ui.button("Export workspace").clicked() {
                ui.close();
                self.export_crossswap();
            }
        });
    }

    fn render_studio_shell(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.heading("noperson extra");
                ui.separator();
                self.workspace_actions(ui);
                ui.separator();
                self.runtime_actions(ui);
                ui.separator();
                ui.label(&self.state.workspace_name);
                if self.state.dirty {
                    ui.label(egui::RichText::new("●").color(egui::Color32::from_rgb(52, 120, 246)));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(&self.state.status).weak());
                });
            });
            ui.separator();

            let available = ui.available_size();
            let left_width = 270.0_f32.min((available.x * 0.24).max(230.0));
            let right_width = 360.0_f32.min((available.x * 0.30).max(300.0));
            let center_width = (available.x - left_width - right_width - 16.0).max(420.0);

            ui.horizontal(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(left_width, available.y),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| panels::media_dock(ui, &mut self.state),
                );
                ui.separator();
                ui.allocate_ui_with_layout(
                    egui::vec2(center_width, available.y),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| panels::player_workspace(ui, &mut self.state),
                );
                ui.separator();
                ui.allocate_ui_with_layout(
                    egui::vec2(right_width, available.y),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| panels::inspector(ui, &mut self.state, &self.models_dir),
                );
            });
        });
    }
}

impl eframe::App for ExtraGuiApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        Self::install_style(ctx);
        self.sync_preview_texture(ctx);
        self.sync_face_textures(ctx);
        self.autosave();
        self.poll_runtime(ctx);
        self.consume_dropped_media(ctx);
        self.consume_analysis_intent();
        self.keyboard_shortcuts(ctx);
        self.consume_timeline_intents();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.render_studio_shell(ui);
    }
}

pub fn launch(models_dir: PathBuf) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([1100.0, 720.0])
            .with_title("noperson extra"),
        ..Default::default()
    };
    eframe::run_native(
        "noperson extra",
        options,
        Box::new(move |cc| {
            Ok(Box::new(ExtraGuiApp::new_with_render_state(
                models_dir,
                cc.wgpu_render_state.clone(),
            )))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable as _};

    #[test]
    fn studio_shell_keeps_media_player_timeline_and_inspector_visible_together() {
        let harness = Harness::builder()
            .with_size(egui::vec2(1440.0, 900.0))
            .build_ui_state(
                |ui, app| app.render_studio_shell(ui),
                ExtraGuiApp::new(PathBuf::from("models")),
            );

        for label in [
            "TARGET MEDIA",
            "SOURCE IDENTITIES",
            "PLAYER",
            "TIMELINE",
            "Faces",
            "Parameters",
            "Settings",
            "Play",
            "Record",
        ] {
            assert!(
                harness.query_by_label(label).is_some(),
                "studio shell is missing {label}"
            );
        }
    }
}
