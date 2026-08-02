use std::collections::{BTreeMap, HashMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

use super::controls::{ControlScope, ControlSpec, ControlState, ControlStateError};
use super::editor::{EditorTimeline, MediaId, MediaLibrary, MediaRole, PreviewViewport};
use super::faces::{EmbeddingStore, FaceCrop, FaceWorkspace, FaceWorkspaceError};

const WORKSPACE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceDocument {
    pub version: u32,
    pub name: String,
    pub media: MediaLibrary,
    #[serde(default)]
    pub faces: FaceWorkspace,
    pub controls: ControlState,
    pub preview: PreviewViewport,
    pub timeline: EditorTimeline,
    #[serde(default)]
    pub output_directory: Option<PathBuf>,
}

impl WorkspaceDocument {
    pub fn new(name: impl Into<String>, catalog: &[ControlSpec]) -> Result<Self, WorkspaceError> {
        Ok(Self {
            version: WORKSPACE_VERSION,
            name: name.into(),
            media: MediaLibrary::default(),
            faces: FaceWorkspace::default(),
            controls: ControlState::from_catalog(catalog)?,
            preview: PreviewViewport::default(),
            timeline: EditorTimeline::default(),
            output_directory: None,
        })
    }

    pub fn save_atomic(&self, path: &Path) -> Result<(), WorkspaceError> {
        let temporary = PathBuf::from(format!("{}.tmp", path.display()));
        let bytes = serde_json::to_vec_pretty(self)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        Ok(())
    }

    pub fn load(path: &Path, catalog: &[ControlSpec]) -> Result<Self, WorkspaceError> {
        let workspace: Self = serde_json::from_slice(&fs::read(path)?)?;
        if workspace.version != WORKSPACE_VERSION {
            return Err(WorkspaceError::UnsupportedVersion(workspace.version));
        }
        workspace.controls.validate_against(catalog)?;
        workspace.faces.validate(catalog)?;
        Ok(workspace)
    }

    pub fn from_crossswap_value(
        name: impl Into<String>,
        value: Value,
        catalog: &[ControlSpec],
    ) -> Result<Self, WorkspaceError> {
        let root = value.as_object().ok_or(WorkspaceError::InvalidCrossSwap(
            "workspace root must be an object",
        ))?;
        let mut workspace = Self::new(name, catalog)?;
        let mut legacy_targets = HashMap::<String, MediaId>::new();

        if let Some(targets) = root.get("target_medias_data").and_then(Value::as_array) {
            for target in targets {
                let Some(target) = target.as_object() else {
                    continue;
                };
                let Some(path) = target.get("media_path").and_then(Value::as_str) else {
                    continue;
                };
                let id = workspace
                    .media
                    .add(MediaRole::Target, PathBuf::from(path))?;
                if let Some(legacy_id) = target.get("media_id").and_then(legacy_id) {
                    legacy_targets.insert(legacy_id, id);
                }
            }
        }
        if let Some(sources) = root.get("input_faces_data").and_then(Value::as_object) {
            for (legacy_id, source) in sources {
                let Some(path) = source.get("media_path").and_then(Value::as_str) else {
                    continue;
                };
                let media_id = workspace
                    .media
                    .add(MediaRole::Source, PathBuf::from(path))?;
                let name = Path::new(path)
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or(legacy_id);
                workspace.faces.add_pending_source(
                    legacy_id.clone(),
                    name.to_owned(),
                    Some(media_id),
                )?;
            }
        }
        if let Some(selected) = root.get("selected_media_id").and_then(legacy_id)
            && let Some(id) = legacy_targets.get(&selected).copied()
        {
            workspace.media.select(MediaRole::Target, id)?;
        }

        if let Some(control) = root.get("control").and_then(Value::as_object) {
            workspace.output_directory = control
                .get("OutputMediaFolder")
                .and_then(Value::as_str)
                .filter(|path| !path.is_empty())
                .map(PathBuf::from);
            workspace.controls.apply_plain_json(control, catalog)?;
        }
        if let Some(parameters) = root
            .get("current_widget_parameters")
            .and_then(Value::as_object)
        {
            workspace.controls.apply_plain_json(parameters, catalog)?;
        }

        if let Some(embeddings) = root.get("embeddings_data").and_then(Value::as_object) {
            for (embedding_id, embedding) in embeddings {
                let Some(embedding) = embedding.as_object() else {
                    continue;
                };
                let store = parse_embedding_store(embedding.get("embedding_store"))?;
                let name = embedding
                    .get("embedding_name")
                    .and_then(Value::as_str)
                    .unwrap_or(embedding_id);
                workspace
                    .faces
                    .add_merged_store(embedding_id.clone(), name.to_owned(), store)?;
            }
        }

        if let Some(targets) = root.get("target_faces_data").and_then(Value::as_object) {
            for (face_id, target) in targets {
                let Some(target) = target.as_object() else {
                    continue;
                };
                let store = parse_embedding_store(target.get("embedding_store"))?;
                let crop = target
                    .get("cropped_face")
                    .filter(|value| !value.is_null())
                    .map(parse_face_crop)
                    .transpose()?;
                workspace.faces.add_target_with_crop(
                    face_id.clone(),
                    None,
                    store,
                    crop,
                    catalog,
                )?;
                let face = workspace
                    .faces
                    .target_mut(face_id)
                    .expect("target was inserted above");
                face.controls = workspace.controls.clone();
                if let Some(control) = target.get("control").and_then(Value::as_object) {
                    face.controls.apply_plain_json(control, catalog)?;
                }
                if let Some(parameters) = target.get("parameters").and_then(Value::as_object) {
                    face.controls.apply_plain_json(parameters, catalog)?;
                }
                face.assigned_embedding_cache =
                    parse_embedding_store_or_empty(target.get("assigned_input_embedding"))?;
                for source_id in string_array(target.get("assigned_input_faces")) {
                    workspace.faces.assign_source(face_id, &source_id, true)?;
                }
                for merged_id in string_array(target.get("assigned_merged_embeddings")) {
                    workspace.faces.assign_merged(face_id, &merged_id, true)?;
                }
            }
        }

        if let Some(markers) = root.get("markers").and_then(Value::as_object) {
            let maximum = markers
                .keys()
                .filter_map(|frame| frame.parse::<u64>().ok())
                .max()
                .unwrap_or(0);
            workspace.timeline = EditorTimeline::new(maximum.saturating_add(1).max(1), 30.0);
            for (frame, marker) in markers {
                let Ok(frame) = frame.parse::<u64>() else {
                    continue;
                };
                let Some(marker) = marker.as_object() else {
                    continue;
                };
                let mut controls = workspace.controls.clone();
                if let Some(control) = marker.get("control").and_then(Value::as_object) {
                    controls.apply_plain_json(control, catalog)?;
                }
                let mut face_controls = BTreeMap::new();
                if let Some(per_face) = marker.get("parameters").and_then(Value::as_object) {
                    for (face_id, parameters) in per_face {
                        let Some(parameters) = parameters.as_object() else {
                            continue;
                        };
                        let mut snapshot = workspace
                            .faces
                            .target(face_id)
                            .map_or_else(|| controls.clone(), |face| face.controls.clone());
                        if let Some(control) = marker.get("control").and_then(Value::as_object) {
                            snapshot.apply_plain_json(control, catalog)?;
                        }
                        snapshot.apply_plain_json(parameters, catalog)?;
                        face_controls.insert(face_id.clone(), snapshot);
                    }
                }
                workspace.timeline.seek(frame);
                workspace
                    .timeline
                    .add_marker_snapshot(controls, face_controls);
            }
            workspace.timeline.seek(0);
        }
        Ok(workspace)
    }

    pub fn to_crossswap_value(&self, catalog: &[ControlSpec]) -> Result<Value, WorkspaceError> {
        self.controls.validate_against(catalog)?;
        self.faces.validate(catalog)?;
        let mut settings = self
            .controls
            .plain_json_for_scope(ControlScope::Settings, catalog);
        settings.insert(
            "OutputMediaFolder".to_owned(),
            self.output_directory.as_ref().map_or_else(
                || Value::String(String::new()),
                |path| Value::String(path.display().to_string()),
            ),
        );
        let mut parameters = self
            .controls
            .plain_json_for_scope(ControlScope::Common, catalog);
        parameters.extend(
            self.controls
                .plain_json_for_scope(ControlScope::Swapper, catalog),
        );

        let targets: Vec<_> = self
            .media
            .items(MediaRole::Target)
            .map(|item| {
                json!({
                    "media_id": item.id.get().to_string(),
                    "media_path": item.path,
                })
            })
            .collect();
        let mut sources = Map::<String, Value>::new();
        let mut referenced_source_media = std::collections::BTreeSet::new();
        for source in self.faces.sources() {
            let Some(media_id) = source.media_id else {
                continue;
            };
            let Some(media) = self.media.item(media_id) else {
                continue;
            };
            referenced_source_media.insert(media_id);
            sources.insert(source.id.clone(), json!({"media_path": media.path}));
        }
        for item in self.media.items(MediaRole::Source) {
            if !referenced_source_media.contains(&item.id) {
                sources.insert(item.id.get().to_string(), json!({"media_path": item.path}));
            }
        }
        let selected = self
            .media
            .selected(MediaRole::Target)
            .map_or(Value::Bool(false), |id| Value::String(id.get().to_string()));

        let markers: Map<String, Value> = self
            .timeline
            .markers()
            .map(|(frame, marker)| {
                let marker_control = marker
                    .controls
                    .plain_json_for_scope(ControlScope::Settings, catalog);
                let mut per_face = Map::new();
                for (face_id, controls) in &marker.face_controls {
                    let mut parameters =
                        controls.plain_json_for_scope(ControlScope::Common, catalog);
                    parameters
                        .extend(controls.plain_json_for_scope(ControlScope::Swapper, catalog));
                    per_face.insert(face_id.clone(), Value::Object(parameters));
                }
                if per_face.is_empty() {
                    let mut parameters = marker
                        .controls
                        .plain_json_for_scope(ControlScope::Common, catalog);
                    parameters.extend(
                        marker
                            .controls
                            .plain_json_for_scope(ControlScope::Swapper, catalog),
                    );
                    per_face.insert("default".to_owned(), Value::Object(parameters));
                }
                (
                    frame.to_string(),
                    json!({
                        "parameters": per_face,
                        "control": marker_control,
                    }),
                )
            })
            .collect();

        let embeddings: Map<String, Value> = self
            .faces
            .merged_identities()
            .map(|identity| {
                (
                    identity.id.clone(),
                    json!({
                        "embedding_name": identity.name,
                        "embedding_store": identity.embeddings,
                    }),
                )
            })
            .collect();
        let target_faces: Map<String, Value> = self
            .faces
            .targets()
            .map(|face| {
                let control = face
                    .controls
                    .plain_json_for_scope(ControlScope::Settings, catalog);
                let mut parameters = face
                    .controls
                    .plain_json_for_scope(ControlScope::Common, catalog);
                parameters.extend(
                    face.controls
                        .plain_json_for_scope(ControlScope::Swapper, catalog),
                );
                (
                    face.id.clone(),
                    json!({
                        "cropped_face": face.crop.as_ref().map(face_crop_value).unwrap_or(Value::Array(Vec::new())),
                        "embedding_store": face.embeddings,
                        "parameters": parameters,
                        "control": control,
                        "assigned_input_faces": face.assigned_sources,
                        "assigned_merged_embeddings": face.assigned_merged,
                        "assigned_input_embedding": face.assigned_embedding_cache,
                    }),
                )
            })
            .collect();

        Ok(json!({
            "selected_media_id": selected,
            "target_medias_data": targets,
            "target_faces_data": target_faces,
            "embeddings_data": embeddings,
            "input_faces_data": sources,
            "markers": markers,
            "control": settings,
            "last_target_media_folder_path": self.media.last_directory(),
            "last_input_media_folder_path": self.media.last_directory(),
            "loaded_embedding_filename": "",
            "current_widget_parameters": parameters,
        }))
    }
}

fn legacy_id(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(false) | Value::Null => None,
        _ => None,
    }
}

fn parse_embedding_store(value: Option<&Value>) -> Result<EmbeddingStore, WorkspaceError> {
    let store = parse_embedding_store_or_empty(value)?;
    if store.is_empty() {
        return Err(WorkspaceError::InvalidCrossSwap(
            "embedding_store must contain at least one model",
        ));
    }
    Ok(store)
}

fn parse_embedding_store_or_empty(value: Option<&Value>) -> Result<EmbeddingStore, WorkspaceError> {
    let Some(store) = value.and_then(Value::as_object) else {
        return Ok(EmbeddingStore::new());
    };
    store
        .iter()
        .map(|(model, embedding)| {
            let values = embedding
                .as_array()
                .ok_or(WorkspaceError::InvalidCrossSwap(
                    "embedding values must be arrays",
                ))?
                .iter()
                .map(|value| {
                    value.as_f64().map(|value| value as f32).ok_or(
                        WorkspaceError::InvalidCrossSwap("embedding values must be numbers"),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((model.clone(), values))
        })
        .collect()
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn parse_face_crop(value: &Value) -> Result<FaceCrop, WorkspaceError> {
    let rows = value.as_array().ok_or(WorkspaceError::InvalidCrossSwap(
        "cropped_face must be an RGB row array",
    ))?;
    let height = rows.len();
    let width = rows.first().and_then(Value::as_array).map_or(0, Vec::len);
    if height == 0 || width == 0 {
        return Err(WorkspaceError::InvalidCrossSwap(
            "cropped_face must not be empty",
        ));
    }
    let mut rgb = Vec::with_capacity(width * height * 3);
    for row in rows {
        let row = row.as_array().ok_or(WorkspaceError::InvalidCrossSwap(
            "cropped_face rows must be arrays",
        ))?;
        if row.len() != width {
            return Err(WorkspaceError::InvalidCrossSwap(
                "cropped_face rows must have equal width",
            ));
        }
        for pixel in row {
            let channels = pixel
                .as_array()
                .filter(|channels| channels.len() == 3)
                .ok_or(WorkspaceError::InvalidCrossSwap(
                    "cropped_face pixels must contain RGB triples",
                ))?;
            for channel in channels {
                let channel = channel.as_u64().filter(|value| *value <= 255).ok_or(
                    WorkspaceError::InvalidCrossSwap("cropped_face channels must be bytes"),
                )?;
                rgb.push(channel as u8);
            }
        }
    }
    Ok(FaceCrop {
        width: width as u32,
        height: height as u32,
        rgb,
    })
}

fn face_crop_value(crop: &FaceCrop) -> Value {
    let rows = crop
        .rgb
        .chunks_exact(crop.width as usize * 3)
        .take(crop.height as usize)
        .map(|row| {
            Value::Array(
                row.chunks_exact(3)
                    .map(|pixel| json!([pixel[0], pixel[1], pixel[2]]))
                    .collect(),
            )
        })
        .collect();
    Value::Array(rows)
}

#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Controls(#[from] ControlStateError),
    #[error(transparent)]
    Faces(#[from] FaceWorkspaceError),
    #[error(transparent)]
    Media(#[from] super::editor::MediaError),
    #[error("unsupported native workspace version {0}")]
    UnsupportedVersion(u32),
    #[error("invalid CrossSwap workspace: {0}")]
    InvalidCrossSwap(&'static str),
}
