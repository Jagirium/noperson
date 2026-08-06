use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui;

use super::controls::{ChoiceSource, ControlSpec, ControlState, control_catalog};
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

pub(super) struct ExtraGuiState {
    pub active_panel: ExtraGuiPanel,
    pub workspace_name: String,
    pub workspace_path: Option<PathBuf>,
    pub dirty: bool,
    pub status: String,
    pub runtime_phase: String,
    pub runtime_progress: Option<(u64, Option<u64>)>,
    pub last_output: Option<PathBuf>,
    pub output_directory: Option<PathBuf>,
    pub catalog: Vec<ControlSpec>,
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
        let catalog = control_catalog().expect("embedded extra GUI control catalog must be valid");
        let controls =
            ControlState::from_catalog(&catalog).expect("control defaults must match their specs");
        let mut dynamic_choices = BTreeMap::new();
        for control in &catalog {
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
