//! Professional CrossSwap-compatible editor frontend.
//!
//! This module intentionally shares backend crates with the realtime app but
//! owns its complete GUI state and lifecycle.

mod app;
mod bridge;
mod controls;
mod editor;
mod faces;
mod panels;
mod runtime;
mod state;
mod workspace;

pub use app::{ExtraGuiApp, launch};
pub use bridge::{ControlBridgeError, EditorEngineRequest, EditorRuntimeConfig};
pub use controls::{
    ChoiceSource, ControlCatalogError, ControlDependency, ControlKind, ControlScope, ControlSpec,
    ControlState, ControlStateError, ControlValue, DependencyMode, DependencyValue, FrontendMode,
    Visibility, control_catalog,
};
pub use editor::{
    EditorMarker, EditorTimeline, MediaError, MediaId, MediaItem, MediaKind, MediaLibrary,
    MediaRole, PreviewViewport, discover_media,
};
pub use faces::{
    ARC_FACE_MODEL, EmbeddingFileError, EmbeddingMergeMethod, EmbeddingStore, FaceCrop,
    FaceWorkspace, FaceWorkspaceError, MergedIdentity, SavedEmbedding, SourceIdentity, TargetFace,
    load_embeddings, save_embeddings,
};
pub use runtime::{EditorJobPhase, EditorJobState, EditorRuntimeError};
pub use state::ExtraGuiPanel;
pub use workspace::{WorkspaceDocument, WorkspaceError};
