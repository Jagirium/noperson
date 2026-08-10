use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;

use super::controls::{ChoiceSource, ControlScope, ControlSpec, ControlState, control_catalog};
use super::editor::{EditorTimeline, MediaId, MediaLibrary, PreviewViewport};
use super::faces::FaceWorkspace;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExtraGuiPanel {
    #[default]
    Media,
    Faces,
    Preview,
    Timeline,
    Parameters,
    Settings,
}

impl ExtraGuiPanel {
    pub const ALL: [Self; 6] = [
        Self::Media,
        Self::Faces,
        Self::Preview,
        Self::Timeline,
        Self::Parameters,
        Self::Settings,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Media => "Media",
            Self::Faces => "Faces",
            Self::Preview => "Preview",
            Self::Timeline => "Timeline",
            Self::Parameters => "Parameters",
            Self::Settings => "Settings",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum InspectorDock {
    Left,
    #[default]
    Right,
    Floating,
}

impl InspectorDock {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Left => "Dock left",
            Self::Right => "Dock right",
            Self::Floating => "Detach inspector",
        }
    }
}

pub(super) struct ExtraGuiState {
    pub active_panel: ExtraGuiPanel,
    pub inspector_scope: ControlScope,
    pub inspector_dock: InspectorDock,
    pub inspector_open: bool,
    pub workspace_name: String,
    pub workspace_path: Option<PathBuf>,
    pub dirty: bool,
    pub status: String,
    pub runtime_phase: String,
    pub runtime_progress: Option<(u64, Option<u64>)>,
    pub last_output: Option<PathBuf>,
    pub output_directory: Option<PathBuf>,
    pub catalog: Arc<[ControlSpec]>,
    pub controls: ControlState,
    pub parameter_search: String,
    pub dynamic_choices: BTreeMap<String, Vec<String>>,
    pub media: MediaLibrary,
    pub target_search: String,
    pub source_search: String,
    pub embedding_search: String,
    pub show_target_images: bool,
    pub show_target_videos: bool,
    pub faces: FaceWorkspace,
    pub face_textures: BTreeMap<String, egui::TextureHandle>,
    pub source_face_textures: BTreeMap<String, egui::TextureHandle>,
    pub merge_selection: BTreeSet<String>,
    pub merge_name: String,
    pub embeddings_path: Option<PathBuf>,
    pub preview: PreviewViewport,
    pub source_texture: Option<egui::TextureHandle>,
    pub source_gpu_texture: Option<egui::load::SizedTexture>,
    pub output_texture: Option<egui::TextureHandle>,
    pub output_gpu_texture: Option<egui::load::SizedTexture>,
    pub latest_output: Option<crate::live::ProcessedRgb>,
    pub preview_loaded: Option<MediaId>,
    pub fullscreen: bool,
    pub timeline: EditorTimeline,
    pub play_requested: bool,
    pub pause_requested: bool,
    pub record_requested: bool,
    pub analysis_requested: bool,
    pub pending_auto_sources: BTreeSet<String>,
    pub pending_auto_merged: BTreeSet<String>,
}

impl Default for ExtraGuiState {
    fn default() -> Self {
        Self::new(Path::new("models"))
    }
}

impl ExtraGuiState {
    pub fn new(models_dir: &Path) -> Self {
        let catalog: Arc<[ControlSpec]> = control_catalog()
            .expect("embedded extra GUI control catalog must be valid")
            .into();
        let controls =
            ControlState::from_catalog(&catalog).expect("control defaults must match their specs");
        let mut dynamic_choices = BTreeMap::new();
        for control in catalog.iter() {
            let super::controls::ControlKind::Choice {
                source: Some(source),
                ..
            } = &control.kind
            else {
                continue;
            };
            let choices = match source {
                ChoiceSource::DfmModels => find_dfm_models(models_dir),
                ChoiceSource::Cameras => vec!["Choose camera".to_owned()],
            };
            dynamic_choices.insert(control.id.clone(), choices);
        }
        Self {
            active_panel: ExtraGuiPanel::Media,
            inspector_scope: ControlScope::Swapper,
            inspector_dock: InspectorDock::Right,
            inspector_open: true,
            workspace_name: "Untitled workspace".to_owned(),
            workspace_path: None,
            dirty: false,
            status: "Editor ready".to_owned(),
            runtime_phase: "Idle".to_owned(),
            runtime_progress: None,
            last_output: None,
            output_directory: None,
            catalog,
            controls,
            parameter_search: String::new(),
            dynamic_choices,
            media: MediaLibrary::default(),
            target_search: String::new(),
            source_search: String::new(),
            embedding_search: String::new(),
            show_target_images: true,
            show_target_videos: true,
            faces: FaceWorkspace::default(),
            face_textures: BTreeMap::new(),
            source_face_textures: BTreeMap::new(),
            merge_selection: BTreeSet::new(),
            merge_name: String::new(),
            embeddings_path: None,
            preview: PreviewViewport::default(),
            source_texture: None,
            source_gpu_texture: None,
            output_texture: None,
            output_gpu_texture: None,
            latest_output: None,
            preview_loaded: None,
            fullscreen: false,
            timeline: EditorTimeline::default(),
            play_requested: false,
            pause_requested: false,
            record_requested: false,
            analysis_requested: false,
            pending_auto_sources: BTreeSet::new(),
            pending_auto_merged: BTreeSet::new(),
        }
    }
}

fn find_dfm_models(models_dir: &Path) -> Vec<String> {
    let mut models = Vec::new();
    collect_dfm_models(models_dir, 2, &mut models);
    models.sort();
    models.dedup();
    models
}

fn collect_dfm_models(dir: &Path, depth: usize, models: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && depth > 0 {
            collect_dfm_models(&path, depth - 1, models);
        } else if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dfm"))
            && let Some(name) = path.file_name().and_then(|name| name.to_str())
        {
            models.push(name.to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspector_dock_defaults_to_open_on_the_right() {
        let state = ExtraGuiState::new(Path::new("models"));

        assert_eq!(state.inspector_dock, InspectorDock::Right);
        assert!(state.inspector_open);
    }

    #[test]
    fn inspector_dock_supports_both_sides_and_a_floating_window() {
        assert_eq!(InspectorDock::Left.label(), "Dock left");
        assert_eq!(InspectorDock::Right.label(), "Dock right");
        assert_eq!(InspectorDock::Floating.label(), "Detach inspector");
    }

    #[test]
    fn control_catalog_is_shared_instead_of_deep_cloned_for_rendering() {
        let state = ExtraGuiState::new(Path::new("models"));
        let catalog = state.catalog.clone();

        assert!(std::sync::Arc::ptr_eq(&catalog, &state.catalog));
    }
}
