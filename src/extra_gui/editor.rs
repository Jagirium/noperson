use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::controls::ControlState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaRole {
    Target,
    Source,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MediaId(u64);

impl MediaId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaItem {
    pub id: MediaId,
    pub role: MediaRole,
    pub kind: MediaKind,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaLibrary {
    items: Vec<MediaItem>,
    selected_target: Option<MediaId>,
    selected_source: Option<MediaId>,
    last_directory: Option<PathBuf>,
    next_id: u64,
}

impl MediaLibrary {
    pub fn add(&mut self, role: MediaRole, path: PathBuf) -> Result<MediaId, MediaError> {
        let kind = classify(&path).ok_or_else(|| MediaError::Unsupported { path: path.clone() })?;
        if role == MediaRole::Source && kind != MediaKind::Image {
            return Err(MediaError::SourceMustBeImage { path });
        }
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            self.last_directory = Some(parent.to_owned());
        }
        if let Some(existing) = self
            .items
            .iter()
            .find(|item| item.role == role && item.path == path)
            .map(|item| item.id)
        {
            self.select(role, existing)?;
            return Ok(existing);
        }
        let id = MediaId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        self.items.push(MediaItem {
            id,
            role,
            kind,
            path,
        });
        self.select(role, id)?;
        Ok(id)
    }

    pub fn items(&self, role: MediaRole) -> impl Iterator<Item = &MediaItem> {
        self.items.iter().filter(move |item| item.role == role)
    }

    pub fn item(&self, id: MediaId) -> Option<&MediaItem> {
        self.items.iter().find(|item| item.id == id)
    }

    pub fn selected(&self, role: MediaRole) -> Option<MediaId> {
        match role {
            MediaRole::Target => self.selected_target,
            MediaRole::Source => self.selected_source,
        }
    }

    pub fn select(&mut self, role: MediaRole, id: MediaId) -> Result<(), MediaError> {
        let item = self.item(id).ok_or(MediaError::UnknownId(id))?;
        if item.role != role {
            return Err(MediaError::WrongRole { id, role });
        }
        match role {
            MediaRole::Target => self.selected_target = Some(id),
            MediaRole::Source => self.selected_source = Some(id),
        }
        Ok(())
    }

    pub fn remove(&mut self, id: MediaId) -> bool {
        let original_len = self.items.len();
        self.items.retain(|item| item.id != id);
        if self.selected_target == Some(id) {
            self.selected_target = None;
        }
        if self.selected_source == Some(id) {
            self.selected_source = None;
        }
        original_len != self.items.len()
    }

    pub fn clear(&mut self, role: MediaRole) -> Vec<MediaId> {
        let removed = self
            .items
            .iter()
            .filter(|item| item.role == role)
            .map(|item| item.id)
            .collect::<Vec<_>>();
        self.items.retain(|item| item.role != role);
        match role {
            MediaRole::Target => self.selected_target = None,
            MediaRole::Source => self.selected_source = None,
        }
        removed
    }

    pub fn last_directory(&self) -> Option<&Path> {
        self.last_directory.as_deref()
    }
}

#[derive(Debug, Error)]
pub enum MediaError {
    #[error("unsupported media file {}", path.display())]
    Unsupported { path: PathBuf },
    #[error("source face must be an image: {}", path.display())]
    SourceMustBeImage { path: PathBuf },
    #[error("unknown media id {0:?}")]
    UnknownId(MediaId),
    #[error("media {id:?} does not belong to {role:?}")]
    WrongRole { id: MediaId, role: MediaRole },
}

fn classify(path: &Path) -> Option<MediaKind> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "jpg" | "jpeg" | "png" | "bmp" | "webp" => Some(MediaKind::Image),
        "mp4" | "avi" | "mkv" | "mov" | "webm" => Some(MediaKind::Video),
        _ => None,
    }
}

pub fn discover_media(
    directory: &Path,
    role: MediaRole,
    recursive: bool,
) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    discover_media_into(directory, role, recursive, &mut paths)?;
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn discover_media_into(
    directory: &Path,
    role: MediaRole,
    recursive: bool,
    paths: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() && recursive {
            discover_media_into(&path, role, true, paths)?;
        } else if file_type.is_file()
            && classify(&path)
                .is_some_and(|kind| role == MediaRole::Target || kind == MediaKind::Image)
        {
            paths.push(path);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreviewViewport {
    zoom: f32,
    pan: [f32; 2],
    anchor: [f32; 2],
    pub compare: bool,
    pub detached: bool,
}

impl PreviewViewport {
    pub const MIN_ZOOM: f32 = 0.1;
    pub const MAX_ZOOM: f32 = 16.0;

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn pan(&self) -> [f32; 2] {
        self.pan
    }

    pub fn anchor(&self) -> [f32; 2] {
        self.anchor
    }

    pub fn zoom_by(&mut self, factor: f32, anchor: [f32; 2]) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let old_zoom = self.zoom;
        self.zoom = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let scale = self.zoom / old_zoom;
        self.pan[0] = anchor[0] - (anchor[0] - self.pan[0]) * scale;
        self.pan[1] = anchor[1] - (anchor[1] - self.pan[1]) * scale;
        self.anchor = anchor;
    }

    pub fn pan_by(&mut self, delta: [f32; 2]) {
        self.pan[0] += delta[0];
        self.pan[1] += delta[1];
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Default for PreviewViewport {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: [0.0, 0.0],
            anchor: [0.0, 0.0],
            compare: false,
            detached: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditorTimeline {
    total_frames: u64,
    current_frame: u64,
    fps: f32,
    playing: bool,
    markers: BTreeMap<u64, EditorMarker>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditorMarker {
    pub controls: ControlState,
    #[serde(default)]
    pub face_controls: BTreeMap<String, ControlState>,
}

impl EditorTimeline {
    pub fn new(total_frames: u64, fps: f32) -> Self {
        Self {
            total_frames,
            current_frame: 0,
            fps,
            playing: false,
            markers: BTreeMap::new(),
        }
    }

    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn toggle_playback(&mut self) {
        self.playing = !self.playing;
    }

    pub fn set_playing(&mut self, playing: bool) {
        self.playing = playing;
    }

    pub fn seek(&mut self, frame: u64) {
        self.current_frame = frame.min(self.total_frames.saturating_sub(1));
    }

    pub fn step(&mut self, delta: i64) {
        self.seek(self.current_frame.saturating_add_signed(delta));
    }

    pub fn add_marker(&mut self, controls: ControlState) {
        self.add_marker_snapshot(controls, BTreeMap::new());
    }

    pub fn add_marker_snapshot(
        &mut self,
        controls: ControlState,
        face_controls: BTreeMap<String, ControlState>,
    ) {
        self.markers.insert(
            self.current_frame,
            EditorMarker {
                controls,
                face_controls,
            },
        );
    }

    pub fn remove_marker(&mut self, frame: u64) -> Option<EditorMarker> {
        self.markers.remove(&frame)
    }

    pub fn clear_markers(&mut self) -> usize {
        let removed = self.markers.len();
        self.markers.clear();
        removed
    }

    pub fn marker_at(&self, frame: u64) -> Option<&EditorMarker> {
        self.markers.get(&frame)
    }

    pub fn markers(&self) -> impl Iterator<Item = (u64, &EditorMarker)> {
        self.markers.iter().map(|(frame, marker)| (*frame, marker))
    }

    pub fn previous_marker(&self) -> Option<u64> {
        self.markers
            .range(..self.current_frame)
            .next_back()
            .map(|(frame, _)| *frame)
    }

    pub fn next_marker(&self) -> Option<u64> {
        self.markers
            .range(self.current_frame.saturating_add(1)..)
            .next()
            .map(|(frame, _)| *frame)
    }
}

impl Default for EditorTimeline {
    fn default() -> Self {
        Self::new(1, 30.0)
    }
}
