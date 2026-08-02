use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::controls::{ControlSpec, ControlState, ControlStateError};
use super::editor::MediaId;

pub const ARC_FACE_MODEL: &str = "Inswapper128ArcFace";
pub type EmbeddingStore = BTreeMap<String, Vec<f32>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedEmbedding {
    pub name: String,
    pub embedding_store: EmbeddingStore,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceCrop {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingMergeMethod {
    Mean,
    Median,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub id: String,
    pub name: String,
    pub media_id: Option<MediaId>,
    pub embeddings: EmbeddingStore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedIdentity {
    pub id: String,
    pub name: String,
    pub source_ids: Vec<String>,
    pub embeddings: EmbeddingStore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetFace {
    pub id: String,
    pub media_id: Option<MediaId>,
    pub embeddings: EmbeddingStore,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop: Option<FaceCrop>,
    pub assigned_sources: BTreeSet<String>,
    pub assigned_merged: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub assigned_embedding_cache: EmbeddingStore,
    pub controls: ControlState,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FaceWorkspace {
    sources: BTreeMap<String, SourceIdentity>,
    merged: BTreeMap<String, MergedIdentity>,
    targets: BTreeMap<String, TargetFace>,
    selected_target: Option<String>,
    swap_all: bool,
    #[serde(default)]
    swap_all_sources: BTreeSet<String>,
    #[serde(default)]
    swap_all_merged: BTreeSet<String>,
}

impl FaceWorkspace {
    pub fn sources(&self) -> impl Iterator<Item = &SourceIdentity> {
        self.sources.values()
    }

    pub fn merged_identities(&self) -> impl Iterator<Item = &MergedIdentity> {
        self.merged.values()
    }

    pub fn targets(&self) -> impl Iterator<Item = &TargetFace> {
        self.targets.values()
    }

    pub fn source(&self, id: &str) -> Option<&SourceIdentity> {
        self.sources.get(id)
    }

    pub fn set_source_embeddings(
        &mut self,
        id: &str,
        embeddings: EmbeddingStore,
    ) -> Result<(), FaceWorkspaceError> {
        validate_store(&embeddings)?;
        let source = self
            .sources
            .get_mut(id)
            .ok_or_else(|| FaceWorkspaceError::UnknownSource(id.to_owned()))?;
        source.embeddings = embeddings;
        Ok(())
    }

    pub fn merged(&self, id: &str) -> Option<&MergedIdentity> {
        self.merged.get(id)
    }

    pub fn target(&self, id: &str) -> Option<&TargetFace> {
        self.targets.get(id)
    }

    pub fn target_mut(&mut self, id: &str) -> Option<&mut TargetFace> {
        self.targets.get_mut(id)
    }

    pub fn selected_target(&self) -> Option<&str> {
        self.selected_target.as_deref()
    }

    pub const fn swap_all(&self) -> bool {
        self.swap_all
    }

    pub fn set_swap_all(&mut self, enabled: bool) {
        self.swap_all = enabled;
        if enabled {
            self.selected_target = None;
        }
    }

    pub fn select_target(&mut self, id: &str) -> Result<(), FaceWorkspaceError> {
        if !self.targets.contains_key(id) {
            return Err(FaceWorkspaceError::UnknownTarget(id.to_owned()));
        }
        self.selected_target = Some(id.to_owned());
        self.swap_all = false;
        Ok(())
    }

    pub fn add_source(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        media_id: Option<MediaId>,
        embeddings: EmbeddingStore,
    ) -> Result<(), FaceWorkspaceError> {
        validate_store(&embeddings)?;
        let id = id.into();
        if self.sources.contains_key(&id) || self.merged.contains_key(&id) {
            return Err(FaceWorkspaceError::DuplicateIdentity(id));
        }
        self.sources.insert(
            id.clone(),
            SourceIdentity {
                id,
                name: name.into(),
                media_id,
                embeddings,
            },
        );
        Ok(())
    }

    pub fn add_pending_source(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        media_id: Option<MediaId>,
    ) -> Result<(), FaceWorkspaceError> {
        let id = id.into();
        if self.sources.contains_key(&id) || self.merged.contains_key(&id) {
            return Err(FaceWorkspaceError::DuplicateIdentity(id));
        }
        self.sources.insert(
            id.clone(),
            SourceIdentity {
                id,
                name: name.into(),
                media_id,
                embeddings: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub fn add_target(
        &mut self,
        id: impl Into<String>,
        media_id: Option<MediaId>,
        embeddings: EmbeddingStore,
        catalog: &[ControlSpec],
    ) -> Result<(), FaceWorkspaceError> {
        self.add_target_with_crop(id, media_id, embeddings, None, catalog)
    }

    pub fn add_target_with_crop(
        &mut self,
        id: impl Into<String>,
        media_id: Option<MediaId>,
        embeddings: EmbeddingStore,
        crop: Option<FaceCrop>,
        catalog: &[ControlSpec],
    ) -> Result<(), FaceWorkspaceError> {
        validate_store(&embeddings)?;
        let id = id.into();
        if self.targets.contains_key(&id) {
            return Err(FaceWorkspaceError::DuplicateTarget(id));
        }
        self.targets.insert(
            id.clone(),
            TargetFace {
                id: id.clone(),
                media_id,
                embeddings,
                crop,
                assigned_sources: BTreeSet::new(),
                assigned_merged: BTreeSet::new(),
                assigned_embedding_cache: BTreeMap::new(),
                controls: ControlState::from_catalog(catalog)?,
            },
        );
        if self.selected_target.is_none() && !self.swap_all {
            self.selected_target = Some(id);
        }
        Ok(())
    }

    pub fn add_merged_store(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        embeddings: EmbeddingStore,
    ) -> Result<(), FaceWorkspaceError> {
        validate_store(&embeddings)?;
        let id = id.into();
        if self.sources.contains_key(&id) || self.merged.contains_key(&id) {
            return Err(FaceWorkspaceError::DuplicateIdentity(id));
        }
        self.merged.insert(
            id.clone(),
            MergedIdentity {
                id,
                name: name.into(),
                source_ids: Vec::new(),
                embeddings,
            },
        );
        Ok(())
    }

    pub fn create_merged(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        source_ids: &[String],
        method: EmbeddingMergeMethod,
    ) -> Result<(), FaceWorkspaceError> {
        let id = id.into();
        if self.sources.contains_key(&id) || self.merged.contains_key(&id) {
            return Err(FaceWorkspaceError::DuplicateIdentity(id));
        }
        if source_ids.is_empty() {
            return Err(FaceWorkspaceError::EmptySelection);
        }
        let stores = source_ids
            .iter()
            .map(|source_id| {
                self.sources
                    .get(source_id)
                    .map(|source| &source.embeddings)
                    .ok_or_else(|| FaceWorkspaceError::UnknownSource(source_id.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let embeddings = merge_stores(&stores, method, EvenMedian::Average)?;
        self.merged.insert(
            id.clone(),
            MergedIdentity {
                id,
                name: name.into(),
                source_ids: source_ids.to_vec(),
                embeddings,
            },
        );
        Ok(())
    }

    pub fn assign_source(
        &mut self,
        target_id: &str,
        source_id: &str,
        assigned: bool,
    ) -> Result<(), FaceWorkspaceError> {
        self.set_source_assignment(Some(target_id), source_id, assigned)
    }

    pub fn set_source_assignment(
        &mut self,
        target_id: Option<&str>,
        source_id: &str,
        assigned: bool,
    ) -> Result<(), FaceWorkspaceError> {
        if !self.sources.contains_key(source_id) {
            return Err(FaceWorkspaceError::UnknownSource(source_id.to_owned()));
        }
        let assigned_sources = if let Some(target_id) = target_id {
            &mut self
                .targets
                .get_mut(target_id)
                .ok_or_else(|| FaceWorkspaceError::UnknownTarget(target_id.to_owned()))?
                .assigned_sources
        } else {
            &mut self.swap_all_sources
        };
        set_membership(assigned_sources, source_id, assigned);
        Ok(())
    }

    pub fn assign_merged(
        &mut self,
        target_id: &str,
        merged_id: &str,
        assigned: bool,
    ) -> Result<(), FaceWorkspaceError> {
        self.set_merged_assignment(Some(target_id), merged_id, assigned)
    }

    pub fn set_merged_assignment(
        &mut self,
        target_id: Option<&str>,
        merged_id: &str,
        assigned: bool,
    ) -> Result<(), FaceWorkspaceError> {
        if !self.merged.contains_key(merged_id) {
            return Err(FaceWorkspaceError::UnknownMerged(merged_id.to_owned()));
        }
        let assigned_merged = if let Some(target_id) = target_id {
            &mut self
                .targets
                .get_mut(target_id)
                .ok_or_else(|| FaceWorkspaceError::UnknownTarget(target_id.to_owned()))?
                .assigned_merged
        } else {
            &mut self.swap_all_merged
        };
        set_membership(assigned_merged, merged_id, assigned);
        Ok(())
    }

    pub fn source_is_assigned(&self, target_id: Option<&str>, source_id: &str) -> bool {
        target_id.map_or_else(
            || self.swap_all_sources.contains(source_id),
            |target_id| {
                self.targets
                    .get(target_id)
                    .is_some_and(|target| target.assigned_sources.contains(source_id))
            },
        )
    }

    pub fn has_assignment(&self, target_id: Option<&str>) -> bool {
        target_id.map_or_else(
            || !self.swap_all_sources.is_empty() || !self.swap_all_merged.is_empty(),
            |target_id| {
                self.targets.get(target_id).is_some_and(|target| {
                    !target.assigned_sources.is_empty() || !target.assigned_merged.is_empty()
                })
            },
        )
    }

    pub fn merged_is_assigned(&self, target_id: Option<&str>, merged_id: &str) -> bool {
        target_id.map_or_else(
            || self.swap_all_merged.contains(merged_id),
            |target_id| {
                self.targets
                    .get(target_id)
                    .is_some_and(|target| target.assigned_merged.contains(merged_id))
            },
        )
    }

    pub fn assigned_embedding(
        &self,
        target_id: &str,
        model: &str,
        method: EmbeddingMergeMethod,
    ) -> Result<Vec<f32>, FaceWorkspaceError> {
        let target = self
            .targets
            .get(target_id)
            .ok_or_else(|| FaceWorkspaceError::UnknownTarget(target_id.to_owned()))?;
        let stores = target
            .assigned_sources
            .iter()
            .filter_map(|id| self.sources.get(id).map(|identity| &identity.embeddings))
            .chain(
                target
                    .assigned_merged
                    .iter()
                    .filter_map(|id| self.merged.get(id).map(|identity| &identity.embeddings)),
            )
            .collect::<Vec<_>>();
        if stores.is_empty()
            && let Some(cached) = target.assigned_embedding_cache.get(model)
        {
            return Ok(cached.clone());
        }
        let merged = merge_stores(&stores, method, EvenMedian::Lower)?;
        merged
            .get(model)
            .cloned()
            .ok_or_else(|| FaceWorkspaceError::MissingModel(model.to_owned()))
    }

    pub fn swap_all_embedding(
        &self,
        model: &str,
        method: EmbeddingMergeMethod,
    ) -> Result<Vec<f32>, FaceWorkspaceError> {
        let stores = self
            .swap_all_sources
            .iter()
            .filter_map(|id| self.sources.get(id).map(|identity| &identity.embeddings))
            .chain(
                self.swap_all_merged
                    .iter()
                    .filter_map(|id| self.merged.get(id).map(|identity| &identity.embeddings)),
            )
            .collect::<Vec<_>>();
        let merged = merge_stores(&stores, method, EvenMedian::Lower)?;
        merged
            .get(model)
            .cloned()
            .ok_or_else(|| FaceWorkspaceError::MissingModel(model.to_owned()))
    }

    pub fn remove_source(&mut self, id: &str) -> bool {
        let removed = self.sources.remove(id).is_some();
        if removed {
            self.merged
                .retain(|_, merged| !merged.source_ids.iter().any(|source| source == id));
            for target in self.targets.values_mut() {
                target.assigned_sources.remove(id);
                target
                    .assigned_merged
                    .retain(|merged| self.merged.contains_key(merged));
            }
            self.swap_all_sources.remove(id);
        }
        removed
    }

    pub fn remove_merged(&mut self, id: &str) -> bool {
        let removed = self.merged.remove(id).is_some();
        if removed {
            for target in self.targets.values_mut() {
                target.assigned_merged.remove(id);
            }
            self.swap_all_merged.remove(id);
        }
        removed
    }

    pub fn clear_merged(&mut self) -> usize {
        let removed = self.merged.len();
        self.merged.clear();
        self.swap_all_merged.clear();
        for target in self.targets.values_mut() {
            target.assigned_merged.clear();
        }
        removed
    }

    pub fn remove_target(&mut self, id: &str) -> bool {
        let removed = self.targets.remove(id).is_some();
        if self.selected_target.as_deref() == Some(id) {
            self.selected_target = self.targets.keys().next().cloned();
        }
        removed
    }

    pub fn clear_targets(&mut self) -> usize {
        let removed = self.targets.len();
        self.targets.clear();
        self.selected_target = None;
        removed
    }

    pub fn remove_targets_for_media(&mut self, media_id: MediaId) -> usize {
        let removed = self
            .targets
            .values()
            .filter(|target| target.media_id == Some(media_id))
            .map(|target| target.id.clone())
            .collect::<Vec<_>>();
        for id in &removed {
            self.remove_target(id);
        }
        removed.len()
    }

    pub fn validate(&self, catalog: &[ControlSpec]) -> Result<(), FaceWorkspaceError> {
        for source in self.sources.values() {
            if !source.embeddings.is_empty() {
                validate_store(&source.embeddings)?;
            }
        }
        for merged in self.merged.values() {
            validate_store(&merged.embeddings)?;
            for source in &merged.source_ids {
                if !self.sources.contains_key(source) {
                    return Err(FaceWorkspaceError::UnknownSource(source.clone()));
                }
            }
        }
        for target in self.targets.values() {
            validate_store(&target.embeddings)?;
            target.controls.validate_against(catalog)?;
            for source in &target.assigned_sources {
                if !self.sources.contains_key(source) {
                    return Err(FaceWorkspaceError::UnknownSource(source.clone()));
                }
            }
            for merged in &target.assigned_merged {
                if !self.merged.contains_key(merged) {
                    return Err(FaceWorkspaceError::UnknownMerged(merged.clone()));
                }
            }
        }
        if let Some(selected) = &self.selected_target
            && !self.targets.contains_key(selected)
        {
            return Err(FaceWorkspaceError::UnknownTarget(selected.clone()));
        }
        for source in &self.swap_all_sources {
            if !self.sources.contains_key(source) {
                return Err(FaceWorkspaceError::UnknownSource(source.clone()));
            }
        }
        for merged in &self.swap_all_merged {
            if !self.merged.contains_key(merged) {
                return Err(FaceWorkspaceError::UnknownMerged(merged.clone()));
            }
        }
        Ok(())
    }
}

pub fn load_embeddings(path: &Path) -> Result<Vec<SavedEmbedding>, EmbeddingFileError> {
    let bytes = std::fs::read(path)?;
    let embeddings: Vec<SavedEmbedding> = serde_json::from_slice(&bytes)?;
    for embedding in &embeddings {
        validate_store(&embedding.embedding_store)?;
    }
    Ok(embeddings)
}

pub fn save_embeddings(
    path: &Path,
    embeddings: &[SavedEmbedding],
) -> Result<(), EmbeddingFileError> {
    for embedding in embeddings {
        validate_store(&embedding.embedding_store)?;
    }
    let bytes = serde_json::to_vec_pretty(embeddings)?;
    let temporary = temporary_path(path);
    std::fs::write(&temporary, bytes)?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "embeddings.json".into());
    name.push(".tmp");
    path.with_file_name(name)
}

fn set_membership(values: &mut BTreeSet<String>, id: &str, assigned: bool) {
    if assigned {
        values.insert(id.to_owned());
    } else {
        values.remove(id);
    }
}

#[derive(Debug, Clone, Copy)]
enum EvenMedian {
    Lower,
    Average,
}

fn merge_stores(
    stores: &[&EmbeddingStore],
    method: EmbeddingMergeMethod,
    even_median: EvenMedian,
) -> Result<EmbeddingStore, FaceWorkspaceError> {
    if stores.is_empty() {
        return Err(FaceWorkspaceError::EmptySelection);
    }
    let models = stores
        .iter()
        .flat_map(|store| store.keys().cloned())
        .collect::<BTreeSet<_>>();
    let mut result = BTreeMap::new();
    for model in models {
        let vectors = stores
            .iter()
            .filter_map(|store| store.get(&model))
            .collect::<Vec<_>>();
        let Some(dimension) = vectors.first().map(|vector| vector.len()) else {
            continue;
        };
        if dimension == 0 || vectors.iter().any(|vector| vector.len() != dimension) {
            return Err(FaceWorkspaceError::DimensionMismatch(model));
        }
        let mut output = Vec::with_capacity(dimension);
        for index in 0..dimension {
            let mut values = vectors
                .iter()
                .map(|vector| vector[index])
                .collect::<Vec<_>>();
            if values.iter().any(|value| !value.is_finite()) {
                return Err(FaceWorkspaceError::NonFinite(model));
            }
            output.push(match method {
                EmbeddingMergeMethod::Mean => values.iter().sum::<f32>() / values.len() as f32,
                EmbeddingMergeMethod::Median => {
                    values.sort_by(f32::total_cmp);
                    let middle = values.len() / 2;
                    if values.len() % 2 == 0 {
                        match even_median {
                            EvenMedian::Lower => values[middle - 1],
                            EvenMedian::Average => (values[middle - 1] + values[middle]) * 0.5,
                        }
                    } else {
                        values[middle]
                    }
                }
            });
        }
        result.insert(model, output);
    }
    if result.is_empty() {
        return Err(FaceWorkspaceError::EmptySelection);
    }
    Ok(result)
}

fn validate_store(store: &EmbeddingStore) -> Result<(), FaceWorkspaceError> {
    if store.is_empty() {
        return Err(FaceWorkspaceError::EmptyEmbeddingStore);
    }
    for (model, embedding) in store {
        if embedding.is_empty() {
            return Err(FaceWorkspaceError::DimensionMismatch(model.clone()));
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(FaceWorkspaceError::NonFinite(model.clone()));
        }
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum FaceWorkspaceError {
    #[error("identity {0} already exists")]
    DuplicateIdentity(String),
    #[error("target face {0} already exists")]
    DuplicateTarget(String),
    #[error("unknown source identity {0}")]
    UnknownSource(String),
    #[error("unknown merged identity {0}")]
    UnknownMerged(String),
    #[error("unknown target face {0}")]
    UnknownTarget(String),
    #[error("no identities selected")]
    EmptySelection,
    #[error("embedding store is empty")]
    EmptyEmbeddingStore,
    #[error("embedding dimensions do not match for {0}")]
    DimensionMismatch(String),
    #[error("embedding for {0} contains a non-finite value")]
    NonFinite(String),
    #[error("no assigned embedding exists for model {0}")]
    MissingModel(String),
    #[error(transparent)]
    Controls(#[from] ControlStateError),
}

#[derive(Debug, Error)]
pub enum EmbeddingFileError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Workspace(#[from] FaceWorkspaceError),
}
