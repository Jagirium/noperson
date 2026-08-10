use std::path::{Path, PathBuf};

use eframe::egui;

use super::controls::{ControlKind, ControlScope, ControlSpec, ControlValue, FrontendMode};
use super::editor::{MediaKind, MediaRole};
use super::faces::{ARC_FACE_MODEL, EmbeddingMergeMethod};
use super::state::{ExtraGuiPanel, ExtraGuiState, InspectorDock};

fn heading(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.heading(title);
    ui.label(egui::RichText::new(subtitle).weak());
    ui.add_space(12.0);
}

fn media_column(ui: &mut egui::Ui, state: &mut ExtraGuiState, role: MediaRole) {
    ui.group(|ui| {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Add files").clicked() {
                let dialog = media_dialog(state, role);
                if let Some(paths) = dialog.pick_files() {
                    add_media_paths(state, role, paths);
                }
            }
            if ui.button("Add folder").clicked() {
                let mut dialog = rfd::FileDialog::new();
                if let Some(directory) = state.media.last_directory() {
                    dialog = dialog.set_directory(directory);
                }
                if let Some(directory) = dialog.pick_folder() {
                    let recursive = recursive_media_selection(state, role);
                    match super::editor::discover_media(&directory, role, recursive) {
                        Ok(paths) => add_media_paths(state, role, paths),
                        Err(error) => state.status = error.to_string(),
                    }
                }
            }
            if ui.button("Remove").clicked()
                && let Some(id) = state.media.selected(role)
            {
                remove_media(state, role, id);
            }
            if ui.button("Clear").clicked() {
                let removed = state.media.clear(role);
                for id in removed {
                    cleanup_media_identity(state, role, id);
                }
                state.dirty = true;
            }
        });
        let query = match role {
            MediaRole::Target => &mut state.target_search,
            MediaRole::Source => &mut state.source_search,
        };
        ui.add(
            egui::TextEdit::singleline(query)
                .hint_text("Filter media…")
                .desired_width(f32::INFINITY),
        );
        if role == MediaRole::Target {
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.show_target_images, "Images");
                ui.checkbox(&mut state.show_target_videos, "Videos");
            });
        }
        ui.separator();
        let selected = state.media.selected(role);
        let query = match role {
            MediaRole::Target => state.target_search.trim().to_lowercase(),
            MediaRole::Source => state.source_search.trim().to_lowercase(),
        };
        let items: Vec<_> = state
            .media
            .items(role)
            .filter(|item| {
                let matches_kind = role == MediaRole::Source
                    || (item.kind == MediaKind::Image && state.show_target_images)
                    || (item.kind == MediaKind::Video && state.show_target_videos);
                let matches_query =
                    query.is_empty() || item.path.to_string_lossy().to_lowercase().contains(&query);
                matches_kind && matches_query
            })
            .cloned()
            .collect();
        if items.is_empty() {
            ui.label(egui::RichText::new("Drop files here or use Add").weak());
        }
        for item in items {
            let icon = match item.kind {
                MediaKind::Image => "IMAGE",
                MediaKind::Video => "VIDEO",
            };
            let name = item
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unnamed");
            if ui
                .selectable_label(selected == Some(item.id), format!("{icon}  {name}"))
                .on_hover_text(item.path.display().to_string())
                .clicked()
            {
                match state.media.select(role, item.id) {
                    Ok(()) => {
                        let changed = selected != Some(item.id);
                        state.dirty |= changed;
                        if changed && role == MediaRole::Target {
                            if auto_swap_enabled(state) {
                                capture_auto_swap_routes(state);
                            } else {
                                state.pending_auto_sources.clear();
                                state.pending_auto_merged.clear();
                            }
                            state.faces.clear_targets();
                            state.timeline = super::EditorTimeline::default();
                            state.output_texture = None;
                            state.output_gpu_texture = None;
                            state.source_gpu_texture = None;
                            state.latest_output = None;
                            if auto_swap_enabled(state) {
                                state.analysis_requested = true;
                            }
                        }
                    }
                    Err(error) => state.status = error.to_string(),
                }
            }
        }
    });
}

pub(super) fn media_dock(ui: &mut egui::Ui, state: &mut ExtraGuiState) {
    ui.set_min_width(250.0);
    ui.label(egui::RichText::new("MEDIA BIN").strong().size(11.0));
    ui.add_space(4.0);
    egui::ScrollArea::vertical()
        .id_salt("extra_media_dock")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.label(egui::RichText::new("TARGET MEDIA").weak().strong());
            media_column(ui, state, MediaRole::Target);
            ui.add_space(10.0);
            ui.label(egui::RichText::new("SOURCE IDENTITIES").weak().strong());
            media_column(ui, state, MediaRole::Source);
            ui.add_space(10.0);
            ui.group(|ui| {
                ui.strong("OUTPUT QUEUE");
                ui.label(format!("State: {}", state.runtime_phase));
                if let Some((processed, total)) = state.runtime_progress {
                    let fraction = total
                        .filter(|total| *total > 0)
                        .map_or(0.0, |total| processed as f32 / total as f32);
                    ui.add(egui::ProgressBar::new(fraction).show_percentage());
                }
                if let Some(path) = &state.last_output {
                    ui.label(
                        egui::RichText::new(path.display().to_string())
                            .small()
                            .weak(),
                    );
                }
            });
        });
}

pub(super) fn player_workspace(ui: &mut egui::Ui, state: &mut ExtraGuiState) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("PLAYER").strong().size(11.0));
        ui.separator();
        let selected = state
            .media
            .selected(MediaRole::Target)
            .and_then(|id| state.media.item(id))
            .and_then(|item| item.path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("No target selected");
        ui.label(egui::RichText::new(selected).weak());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(&state.runtime_phase).weak());
        });
    });
    ui.add_space(4.0);
    preview_toolbar(ui, state);
    ui.add_space(4.0);

    let timeline_height = 126.0;
    let canvas_height = (ui.available_height() - timeline_height).max(280.0);
    let canvas_width = ui.available_width();
    preview_canvas(ui, state, egui::vec2(canvas_width, canvas_height));
    ui.add_space(6.0);
    compact_timeline(ui, state);

    if state.preview.detached {
        let mut open = true;
        egui::Window::new("Detached preview")
            .open(&mut open)
            .default_size(egui::vec2(960.0, 640.0))
            .show(ui.ctx(), |ui| {
                preview_canvas(ui, state, ui.available_size());
            });
        state.preview.detached = open;
    }
}

pub(super) fn inspector(ui: &mut egui::Ui, state: &mut ExtraGuiState, models_dir: &Path) {
    if !matches!(
        state.active_panel,
        ExtraGuiPanel::Faces | ExtraGuiPanel::Parameters | ExtraGuiPanel::Settings
    ) {
        state.active_panel = ExtraGuiPanel::Faces;
    }
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("INSPECTOR").strong().size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Hide inspector").clicked() {
                state.inspector_open = false;
            }
            match state.inspector_dock {
                InspectorDock::Left => {
                    if ui.small_button(InspectorDock::Right.label()).clicked() {
                        state.inspector_dock = InspectorDock::Right;
                    }
                    if ui.small_button(InspectorDock::Floating.label()).clicked() {
                        state.inspector_dock = InspectorDock::Floating;
                    }
                }
                InspectorDock::Right => {
                    if ui.small_button(InspectorDock::Left.label()).clicked() {
                        state.inspector_dock = InspectorDock::Left;
                    }
                    if ui.small_button(InspectorDock::Floating.label()).clicked() {
                        state.inspector_dock = InspectorDock::Floating;
                    }
                }
                InspectorDock::Floating => {
                    if ui.small_button(InspectorDock::Right.label()).clicked() {
                        state.inspector_dock = InspectorDock::Right;
                    }
                    if ui.small_button(InspectorDock::Left.label()).clicked() {
                        state.inspector_dock = InspectorDock::Left;
                    }
                }
            }
        });
    });
    ui.horizontal_wrapped(|ui| {
        if ui
            .selectable_label(state.active_panel == ExtraGuiPanel::Faces, "Faces")
            .clicked()
        {
            state.active_panel = ExtraGuiPanel::Faces;
        }
        for (scope, label) in [
            (ControlScope::Swapper, "Face Swap"),
            (ControlScope::Common, "Common"),
        ] {
            let selected =
                state.active_panel == ExtraGuiPanel::Parameters && state.inspector_scope == scope;
            if ui.selectable_label(selected, label).clicked() {
                state.active_panel = ExtraGuiPanel::Parameters;
                state.inspector_scope = scope;
            }
        }
        if ui
            .selectable_label(state.active_panel == ExtraGuiPanel::Settings, "Settings")
            .clicked()
        {
            state.active_panel = ExtraGuiPanel::Settings;
        }
    });
    ui.separator();
    match state.active_panel {
        ExtraGuiPanel::Faces => {
            egui::ScrollArea::vertical()
                .id_salt("extra_faces_inspector")
                .auto_shrink([false, false])
                .show(ui, |ui| faces(ui, state));
        }
        ExtraGuiPanel::Parameters => parameters(ui, state, state.inspector_scope),
        ExtraGuiPanel::Settings => {
            egui::ScrollArea::vertical()
                .id_salt("extra_settings_inspector")
                .auto_shrink([false, false])
                .show(ui, |ui| settings(ui, state, models_dir));
        }
        _ => unreachable!("inspector panel was normalized above"),
    }
}

fn media_dialog(state: &ExtraGuiState, role: MediaRole) -> rfd::FileDialog {
    let mut dialog = rfd::FileDialog::new();
    if let Some(directory) = state.media.last_directory() {
        dialog = dialog.set_directory(directory);
    }
    match role {
        MediaRole::Target => dialog.add_filter(
            "Media",
            &[
                "jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mkv", "mov", "webm",
            ],
        ),
        MediaRole::Source => {
            dialog.add_filter("Face images", &["jpg", "jpeg", "png", "bmp", "webp"])
        }
    }
}

fn recursive_media_selection(state: &ExtraGuiState, role: MediaRole) -> bool {
    let control = match role {
        MediaRole::Target => "TargetMediaFolderRecursiveToggle",
        MediaRole::Source => "InputFacesFolderRecursiveToggle",
    };
    matches!(
        state.controls.get(control),
        Some(ControlValue::Toggle(true))
    )
}

pub(super) fn add_media_paths(state: &mut ExtraGuiState, role: MediaRole, paths: Vec<PathBuf>) {
    if role == MediaRole::Target {
        if auto_swap_enabled(state) {
            capture_auto_swap_routes(state);
        } else {
            state.pending_auto_sources.clear();
            state.pending_auto_merged.clear();
        }
    }
    let mut added = 0_usize;
    for path in paths {
        let source_name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("Source face")
            .to_owned();
        match state.media.add(role, path) {
            Ok(media_id) => {
                if role == MediaRole::Source {
                    let identity_id = format!("source-{}", media_id.get());
                    if state.faces.source(&identity_id).is_none()
                        && let Err(error) =
                            state
                                .faces
                                .add_pending_source(identity_id, source_name, Some(media_id))
                    {
                        state.status = error.to_string();
                        continue;
                    }
                }
                if role == MediaRole::Target && auto_swap_enabled(state) {
                    state.analysis_requested = true;
                }
                added += 1;
            }
            Err(error) => state.status = error.to_string(),
        }
    }
    if added > 0 {
        if role == MediaRole::Target {
            state.faces.clear_targets();
            state.timeline = super::EditorTimeline::default();
            state.output_texture = None;
            state.output_gpu_texture = None;
            state.source_gpu_texture = None;
            state.latest_output = None;
        }
        state.status = format!("Added {added} media item(s)");
        state.dirty = true;
    }
}

fn auto_swap_enabled(state: &ExtraGuiState) -> bool {
    matches!(
        state.controls.get("AutoSwapToggle"),
        Some(ControlValue::Toggle(true))
    )
}

fn capture_auto_swap_routes(state: &mut ExtraGuiState) {
    let active_target = if state.faces.swap_all() {
        None
    } else {
        state.faces.selected_target().map(str::to_owned)
    };
    state.pending_auto_sources = state
        .faces
        .sources()
        .filter(|source| {
            state
                .faces
                .source_is_assigned(active_target.as_deref(), &source.id)
        })
        .map(|source| source.id.clone())
        .collect();
    state.pending_auto_merged = state
        .faces
        .merged_identities()
        .filter(|identity| {
            state
                .faces
                .merged_is_assigned(active_target.as_deref(), &identity.id)
        })
        .map(|identity| identity.id.clone())
        .collect();
}

fn cleanup_media_identity(state: &mut ExtraGuiState, role: MediaRole, id: super::MediaId) {
    match role {
        MediaRole::Target => {
            state.faces.remove_targets_for_media(id);
        }
        MediaRole::Source => {
            state.faces.remove_source(&format!("source-{}", id.get()));
        }
    }
}

fn remove_media(state: &mut ExtraGuiState, role: MediaRole, id: super::MediaId) {
    cleanup_media_identity(state, role, id);
    if state.media.remove(id) {
        state.status = "Removed media item".to_owned();
        state.dirty = true;
    }
}

fn faces(ui: &mut egui::Ui, state: &mut ExtraGuiState) {
    heading(
        ui,
        "Face assignments",
        "Detected targets, source identities and per-face routing.",
    );
    let mut swap_all = state.faces.swap_all();
    ui.horizontal(|ui| {
        if ui
            .checkbox(&mut swap_all, "Swap every detected face")
            .changed()
        {
            state.faces.set_swap_all(swap_all);
            state.dirty = true;
        }
        ui.separator();
        let route = if swap_all {
            "Assignments below are the unscoped, final engine route"
        } else if state.faces.selected_target().is_some() {
            "Assignments and parameters belong only to the selected face"
        } else {
            "Load or import detected target faces"
        };
        ui.label(egui::RichText::new(route).weak());
    });
    ui.add_space(8.0);

    ui.columns(3, |columns| {
        columns[0].group(|ui| {
            ui.strong("Detected targets");
            ui.label(egui::RichText::new("ArcFace identity routes").weak());
            ui.separator();
            let targets = state
                .faces
                .targets()
                .map(|face| {
                    (
                        face.id.clone(),
                        face.embeddings.len(),
                        state.face_textures.get(&face.id).cloned(),
                    )
                })
                .collect::<Vec<_>>();
            if targets.is_empty() {
                ui.label(egui::RichText::new("No detected faces yet").weak());
            }
            for (id, models, texture) in targets {
                let selected = !swap_all && state.faces.selected_target() == Some(id.as_str());
                let mut clicked = false;
                let mut remove = false;
                ui.horizontal(|ui| {
                    if let Some(texture) = texture {
                        ui.add(
                            egui::Image::new((texture.id(), egui::vec2(54.0, 54.0)))
                                .corner_radius(6.0),
                        );
                    }
                    clicked = ui
                        .selectable_label(selected, format!("{id}\n{models} model(s)"))
                        .clicked();
                    remove = ui
                        .small_button("×")
                        .on_hover_text("Remove target face")
                        .clicked();
                });
                if remove {
                    state.faces.remove_target(&id);
                    state.dirty = true;
                    continue;
                }
                if clicked {
                    match state.faces.select_target(&id) {
                        Ok(()) => state.dirty = true,
                        Err(error) => state.status = error.to_string(),
                    }
                }
            }
        });

        columns[1].group(|ui| {
            ui.strong("Source identities");
            ui.label(egui::RichText::new("Route into the active target").weak());
            ui.separator();
            let active_target = if swap_all {
                None
            } else {
                state.faces.selected_target().map(str::to_owned)
            };
            let sources = state
                .faces
                .sources()
                .map(|source| {
                    (
                        source.id.clone(),
                        source.name.clone(),
                        source.embeddings.contains_key(ARC_FACE_MODEL),
                        state.source_face_textures.get(&source.id).cloned(),
                    )
                })
                .collect::<Vec<_>>();
            if sources.is_empty() {
                ui.label(egui::RichText::new("Add source images in Media").weak());
            }
            for (id, name, ready, texture) in sources {
                let mut assigned = state
                    .faces
                    .source_is_assigned(active_target.as_deref(), &id);
                let suffix = if ready { "ready" } else { "awaiting scan" };
                let mut remove = false;
                ui.horizontal(|ui| {
                    if let Some(texture) = texture {
                        ui.add(
                            egui::Image::new((texture.id(), egui::vec2(42.0, 42.0)))
                                .corner_radius(6.0),
                        );
                    }
                    if ui
                        .checkbox(&mut assigned, format!("{name} · {suffix}"))
                        .changed()
                    {
                        match state.faces.set_source_assignment(
                            active_target.as_deref(),
                            &id,
                            assigned,
                        ) {
                            Ok(()) => state.dirty = true,
                            Err(error) => state.status = error.to_string(),
                        }
                    }
                    remove = ui
                        .small_button("×")
                        .on_hover_text("Remove source")
                        .clicked();
                });
                if remove {
                    let media_id = state.faces.source(&id).and_then(|source| source.media_id);
                    state.faces.remove_source(&id);
                    if let Some(media_id) = media_id {
                        state.media.remove(media_id);
                    }
                    state.merge_selection.remove(&id);
                    state.dirty = true;
                    continue;
                }
                let mut selected_for_merge = state.merge_selection.contains(&id);
                if ui
                    .checkbox(&mut selected_for_merge, "include in saved embedding")
                    .changed()
                {
                    if selected_for_merge {
                        state.merge_selection.insert(id);
                    } else {
                        state.merge_selection.remove(&id);
                    }
                }
            }
        });

        columns[2].group(|ui| {
            ui.strong("Reusable embeddings");
            ui.horizontal_wrapped(|ui| {
                if ui.button("Load").clicked() {
                    load_embedding_file(state);
                }
                if ui.button("Save").clicked() {
                    save_embedding_file(state, false);
                }
                if ui.button("Save as").clicked() {
                    save_embedding_file(state, true);
                }
                if ui.button("Clear").clicked() {
                    state.faces.clear_merged();
                    state.dirty = true;
                }
            });
            ui.add(
                egui::TextEdit::singleline(&mut state.embedding_search)
                    .hint_text("Filter embeddings…")
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut state.merge_name)
                        .hint_text("Embedding name")
                        .desired_width(150.0),
                );
                if ui.button("Merge").clicked() {
                    let method = merge_method(&state.controls);
                    let mut sequence = state.faces.merged_identities().count() + 1;
                    let id = loop {
                        let candidate = format!("merged-{sequence}");
                        if state.faces.merged(&candidate).is_none() {
                            break candidate;
                        }
                        sequence += 1;
                    };
                    let source_ids = state.merge_selection.iter().cloned().collect::<Vec<_>>();
                    let name = if state.merge_name.trim().is_empty() {
                        id.clone()
                    } else {
                        state.merge_name.trim().to_owned()
                    };
                    match state.faces.create_merged(id, name, &source_ids, method) {
                        Ok(()) => {
                            state.merge_selection.clear();
                            state.merge_name.clear();
                            state.status = "Created reusable embedding".to_owned();
                            state.dirty = true;
                        }
                        Err(error) => state.status = error.to_string(),
                    }
                }
            });
            ui.separator();
            let active_target = if swap_all {
                None
            } else {
                state.faces.selected_target().map(str::to_owned)
            };
            let query = state.embedding_search.trim().to_lowercase();
            let merged = state
                .faces
                .merged_identities()
                .filter(|identity| {
                    query.is_empty()
                        || identity.name.to_lowercase().contains(&query)
                        || identity.id.to_lowercase().contains(&query)
                })
                .map(|identity| (identity.id.clone(), identity.name.clone()))
                .collect::<Vec<_>>();
            if merged.is_empty() {
                ui.label(egui::RichText::new("No saved embeddings").weak());
            }
            for (id, name) in merged {
                let mut assigned = state
                    .faces
                    .merged_is_assigned(active_target.as_deref(), &id);
                let mut remove = false;
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut assigned, name).changed() {
                        match state.faces.set_merged_assignment(
                            active_target.as_deref(),
                            &id,
                            assigned,
                        ) {
                            Ok(()) => state.dirty = true,
                            Err(error) => state.status = error.to_string(),
                        }
                    }
                    remove = ui
                        .small_button("×")
                        .on_hover_text("Remove embedding")
                        .clicked();
                });
                if remove {
                    state.faces.remove_merged(&id);
                    state.dirty = true;
                }
            }
        });
    });
}

fn merge_method(controls: &super::controls::ControlState) -> EmbeddingMergeMethod {
    match controls.get("EmbMergeMethodSelection") {
        Some(ControlValue::Choice(value)) if value == "Median" => EmbeddingMergeMethod::Median,
        _ => EmbeddingMergeMethod::Mean,
    }
}

fn load_embedding_file(state: &mut ExtraGuiState) {
    let mut dialog = rfd::FileDialog::new().add_filter("CrossSwap embeddings", &["json"]);
    if let Some(path) = &state.embeddings_path
        && let Some(directory) = path.parent()
    {
        dialog = dialog.set_directory(directory);
    }
    let Some(path) = dialog.pick_file() else {
        return;
    };
    match super::faces::load_embeddings(&path) {
        Ok(embeddings) => {
            state.faces.clear_merged();
            for (index, embedding) in embeddings.into_iter().enumerate() {
                if let Err(error) = state.faces.add_merged_store(
                    format!("loaded-{index}"),
                    embedding.name,
                    embedding.embedding_store,
                ) {
                    state.status = error.to_string();
                    return;
                }
            }
            state.embeddings_path = Some(path.clone());
            state.status = format!("Loaded embeddings from {}", path.display());
            state.dirty = true;
        }
        Err(error) => state.status = error.to_string(),
    }
}

fn save_embedding_file(state: &mut ExtraGuiState, force_dialog: bool) {
    let path = (!force_dialog)
        .then(|| state.embeddings_path.clone())
        .flatten()
        .or_else(|| {
            rfd::FileDialog::new()
                .add_filter("CrossSwap embeddings", &["json"])
                .set_file_name("embeddings.json")
                .save_file()
        });
    let Some(path) = path else {
        return;
    };
    let embeddings = state
        .faces
        .merged_identities()
        .map(|identity| super::faces::SavedEmbedding {
            name: identity.name.clone(),
            embedding_store: identity.embeddings.clone(),
        })
        .collect::<Vec<_>>();
    match super::faces::save_embeddings(&path, &embeddings) {
        Ok(()) => {
            state.embeddings_path = Some(path.clone());
            state.status = format!("Saved embeddings to {}", path.display());
        }
        Err(error) => state.status = error.to_string(),
    }
}

fn preview_toolbar(ui: &mut egui::Ui, state: &mut ExtraGuiState) {
    ui.horizontal(|ui| {
        if ui.button("−").on_hover_text("Zoom out").clicked() {
            state.preview.zoom_by(0.8, [0.0, 0.0]);
            state.dirty = true;
        }
        ui.label(format!("{:.0}%", state.preview.zoom() * 100.0));
        if ui.button("+").on_hover_text("Zoom in").clicked() {
            state.preview.zoom_by(1.25, [0.0, 0.0]);
            state.dirty = true;
        }
        if ui.button("Fit").clicked() {
            state.preview.reset();
            state.dirty = true;
        }
        if ui
            .toggle_value(&mut state.preview.compare, "Compare")
            .changed()
        {
            state.dirty = true;
        }
        if ui.button("Detach").clicked() {
            state.preview.detached = true;
            state.dirty = true;
        }
        if ui.button("Fullscreen").clicked() {
            state.fullscreen = !state.fullscreen;
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Fullscreen(state.fullscreen));
        }
        if ui.button("Save frame").clicked() {
            save_preview_frame(state);
        }
        ui.label(egui::RichText::new("Wheel to zoom · drag to pan · double-click to fit").weak());
    });
}

fn preview_canvas(ui: &mut egui::Ui, state: &mut ExtraGuiState, size: egui::Vec2) {
    let size = size.max(egui::vec2(320.0, 240.0));
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
    ui.painter()
        .rect_filled(rect, 8.0, egui::Color32::from_rgb(7, 8, 11));
    if response.double_clicked() {
        state.preview.reset();
        state.dirty = true;
    }
    if response.dragged() {
        let delta = ui.input(|input| input.pointer.delta());
        state.preview.pan_by([delta.x, delta.y]);
        state.dirty = true;
    }
    if response.hovered() {
        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let pointer = response.hover_pos().unwrap_or(rect.center()) - rect.center();
            state
                .preview
                .zoom_by((scroll * 0.0025).exp(), [pointer.x, pointer.y]);
            state.dirty = true;
            ui.input_mut(|input| input.smooth_scroll_delta = egui::Vec2::ZERO);
        }
    }
    let source = state.source_gpu_texture.or_else(|| {
        state
            .source_texture
            .as_ref()
            .map(egui::load::SizedTexture::from_handle)
    });
    let output = state.output_gpu_texture.or_else(|| {
        state
            .output_texture
            .as_ref()
            .map(egui::load::SizedTexture::from_handle)
    });
    let active = output.or(source);
    if let Some(texture) = active {
        if state.preview.compare
            && let (Some(source), Some(output)) = (source, output)
        {
            let split = rect.center().x;
            paint_preview_texture(
                ui,
                source,
                rect,
                state,
                egui::Rect::from_min_max(rect.min, egui::pos2(split, rect.max.y)),
            );
            paint_preview_texture(
                ui,
                output,
                rect,
                state,
                egui::Rect::from_min_max(egui::pos2(split, rect.min.y), rect.max),
            );
            ui.painter().line_segment(
                [
                    egui::pos2(split, rect.top()),
                    egui::pos2(split, rect.bottom()),
                ],
                egui::Stroke::new(2.0, egui::Color32::WHITE),
            );
        } else {
            paint_preview_texture(ui, texture, rect, state, rect);
        }
        paint_face_diagnostics(ui, state, rect);
        let overlay = egui::Rect::from_min_size(
            rect.right_top() + egui::vec2(-250.0, 10.0),
            egui::vec2(240.0, 30.0),
        );
        ui.painter().rect(
            overlay,
            7.0,
            egui::Color32::from_black_alpha(190),
            egui::Stroke::new(1.0, egui::Color32::from_white_alpha(28)),
            egui::StrokeKind::Inside,
        );
        ui.painter().text(
            overlay.center(),
            egui::Align2::CENTER_CENTER,
            format!("● GPU · {}", state.runtime_phase),
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(158, 235, 197),
        );
    } else {
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "NO FRAME LOADED",
            egui::FontId::proportional(14.0),
            egui::Color32::from_gray(110),
        );
    }
}

fn paint_preview_texture(
    ui: &egui::Ui,
    texture: egui::load::SizedTexture,
    canvas: egui::Rect,
    state: &ExtraGuiState,
    clip: egui::Rect,
) {
    let image_rect = preview_image_rect(texture, canvas, state);
    ui.painter().with_clip_rect(clip).image(
        texture.id,
        image_rect,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}

fn preview_image_rect(
    texture: egui::load::SizedTexture,
    canvas: egui::Rect,
    state: &ExtraGuiState,
) -> egui::Rect {
    let image = texture.size;
    let fit = (canvas.width() / image.x).min(canvas.height() / image.y);
    let draw_size = image * fit * state.preview.zoom();
    let pan = state.preview.pan();
    egui::Rect::from_center_size(canvas.center() + egui::vec2(pan[0], pan[1]), draw_size)
}

fn paint_face_diagnostics(ui: &egui::Ui, state: &ExtraGuiState, canvas: egui::Rect) {
    let show_boxes = matches!(
        state.controls.get("ShowAllDetectedFacesBBoxToggle"),
        Some(ControlValue::Toggle(true))
    );
    let show_landmarks = matches!(
        state.controls.get("ShowLandmarksEnableToggle"),
        Some(ControlValue::Toggle(true))
    );
    if !show_boxes && !show_landmarks {
        return;
    }
    let texture = state.output_gpu_texture.or_else(|| {
        state
            .output_texture
            .as_ref()
            .map(egui::load::SizedTexture::from_handle)
    });
    let (Some(texture), Some(output)) = (texture, &state.latest_output) else {
        return;
    };
    let image_rect = preview_image_rect(texture, canvas, state);
    let clip = if state.preview.compare {
        egui::Rect::from_min_max(egui::pos2(canvas.center().x, canvas.top()), canvas.max)
    } else {
        canvas
    };
    let painter = ui.painter().with_clip_rect(clip);
    for overlay in &output.overlays {
        if show_boxes {
            let bbox = egui::Rect::from_min_max(
                egui::pos2(
                    image_rect.left() + overlay.bbox[0] * image_rect.width(),
                    image_rect.top() + overlay.bbox[1] * image_rect.height(),
                ),
                egui::pos2(
                    image_rect.left() + overlay.bbox[2] * image_rect.width(),
                    image_rect.top() + overlay.bbox[3] * image_rect.height(),
                ),
            );
            painter.rect_stroke(
                bbox,
                3.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(52, 120, 246)),
                egui::StrokeKind::Inside,
            );
        }
        if show_landmarks {
            for point in overlay.kps_5 {
                painter.circle_filled(
                    egui::pos2(
                        image_rect.left() + point[0] * image_rect.width(),
                        image_rect.top() + point[1] * image_rect.height(),
                    ),
                    3.0,
                    egui::Color32::from_rgb(74, 214, 156),
                );
            }
        }
    }
}

fn save_preview_frame(state: &mut ExtraGuiState) {
    let Some(output) = &state.latest_output else {
        state.status = "Render a preview before saving a frame".to_owned();
        return;
    };
    let mut dialog = rfd::FileDialog::new()
        .add_filter("PNG image", &["png"])
        .set_file_name("noperson-frame.png");
    if let Some(directory) = &state.output_directory {
        dialog = dialog.set_directory(directory);
    }
    let Some(path) = dialog.save_file() else {
        return;
    };
    match image::save_buffer(
        &path,
        &output.data,
        output.width,
        output.height,
        image::ColorType::Rgb8,
    ) {
        Ok(()) => {
            state.output_directory = path.parent().map(PathBuf::from);
            state.status = format!("Saved {}", path.display());
        }
        Err(error) => state.status = error.to_string(),
    }
}

fn compact_timeline(ui: &mut egui::Ui, state: &mut ExtraGuiState) {
    ui.label(egui::RichText::new("TIMELINE").strong().size(11.0));

    let mut frame = state.timeline.current_frame();
    if ui
        .add(
            egui::Slider::new(
                &mut frame,
                0..=state.timeline.total_frames().saturating_sub(1),
            )
            .show_value(false),
        )
        .changed()
    {
        state.timeline.seek(frame);
        apply_exact_marker(state);
        state.dirty = true;
    }

    ui.horizontal_wrapped(|ui| {
        if ui.button("First").clicked() {
            state.timeline.seek(0);
            apply_exact_marker(state);
        }
        if ui.button("Previous frame").clicked() {
            state.timeline.step(-1);
            apply_exact_marker(state);
        }
        let playback_label = if state.timeline.is_playing() {
            "Pause"
        } else {
            "Play"
        };
        if ui.button(playback_label).clicked() {
            state.timeline.toggle_playback();
            if state.timeline.is_playing() {
                state.play_requested = true;
            } else {
                state.pause_requested = true;
            }
        }
        if ui.button("Next frame").clicked() {
            state.timeline.step(1);
            apply_exact_marker(state);
        }
        if ui.button("Last").clicked() {
            state
                .timeline
                .seek(state.timeline.total_frames().saturating_sub(1));
            apply_exact_marker(state);
        }
        if ui.button("Record").clicked() {
            state.record_requested = true;
        }
        ui.separator();
        if ui.button("Add marker").clicked() {
            let face_controls = state
                .faces
                .targets()
                .map(|face| (face.id.clone(), face.controls.clone()))
                .collect();
            state
                .timeline
                .add_marker_snapshot(state.controls.clone(), face_controls);
            state.status = format!("Marker at frame {}", state.timeline.current_frame());
            state.dirty = true;
        }
        if ui.button("Previous marker").clicked()
            && let Some(frame) = state.timeline.previous_marker()
        {
            state.timeline.seek(frame);
            apply_exact_marker(state);
        }
        if ui.button("Next marker").clicked()
            && let Some(frame) = state.timeline.next_marker()
        {
            state.timeline.seek(frame);
            apply_exact_marker(state);
        }
        if ui.button("Remove marker").clicked() {
            state.timeline.remove_marker(state.timeline.current_frame());
            state.dirty = true;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(format!(
                "{} / {} · {:.2} FPS",
                state.timeline.current_frame(),
                state.timeline.total_frames().saturating_sub(1),
                state.timeline.fps()
            ));
        });
    });
}

fn apply_exact_marker(state: &mut ExtraGuiState) {
    let Some(marker) = state
        .timeline
        .marker_at(state.timeline.current_frame())
        .cloned()
    else {
        return;
    };
    state.controls = marker.controls;
    for (face_id, controls) in marker.face_controls {
        if let Some(face) = state.faces.target_mut(&face_id) {
            face.controls = controls;
        }
    }
    state.status = format!("Loaded marker at frame {}", state.timeline.current_frame());
}

fn parameters(ui: &mut egui::Ui, state: &mut ExtraGuiState, scope: ControlScope) {
    let (title, subtitle) = match scope {
        ControlScope::Swapper => (
            "Face swap",
            "Swapper, similarity, landmarks, masks, color and blend controls.",
        ),
        ControlScope::Common => (
            "Common",
            "Face restoration and shared post-processing controls.",
        ),
        ControlScope::Settings => unreachable!("settings have a dedicated inspector tab"),
    };
    heading(ui, title, subtitle);
    ui.horizontal(|ui| {
        ui.label("Filter");
        ui.add(
            egui::TextEdit::singleline(&mut state.parameter_search)
                .hint_text("mask, color, restorer…")
                .desired_width(ui.available_width()),
        );
    });
    let target = state.faces.selected_target().unwrap_or("swap-all defaults");
    ui.label(egui::RichText::new(format!("Editing {target}")).weak());
    ui.add_space(8.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            render_scope(ui, state, scope);
        });
}

fn settings(ui: &mut egui::Ui, state: &mut ExtraGuiState, models_dir: &Path) {
    heading(
        ui,
        "Application settings",
        "GPU provider, detector, webcam, output and workspace behavior.",
    );
    egui::Grid::new("extra_gui_settings_summary")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("Models");
            ui.label(models_dir.display().to_string());
            ui.end_row();
            ui.label("Runtime");
            ui.label("GPU only");
            ui.end_row();
        });
    ui.horizontal(|ui| {
        ui.label("Output folder");
        ui.label(state.output_directory.as_ref().map_or_else(
            || "Choose per export".to_owned(),
            |path| path.display().to_string(),
        ));
        if ui.button("Choose…").clicked()
            && let Some(path) = rfd::FileDialog::new().pick_folder()
        {
            state.output_directory = Some(path);
            state.dirty = true;
        }
    });
    ui.add_space(10.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| render_scope(ui, state, ControlScope::Settings));
}

fn render_scope(ui: &mut egui::Ui, state: &mut ExtraGuiState, scope: ControlScope) {
    let query = state.parameter_search.trim();
    let query = (!query.is_empty()).then(|| query.to_lowercase());
    let values = scoped_control_state(state, scope);
    let catalog = std::sync::Arc::clone(&state.catalog);
    let controls: Vec<_> = catalog
        .iter()
        .filter(|control| control.scope == scope)
        .filter(|control| values.is_spec_visible(control, FrontendMode::Editor))
        .filter(|control| {
            query.as_ref().is_none_or(|query| {
                control.label.to_lowercase().contains(query)
                    || control.section.to_lowercase().contains(query)
                    || control.id.to_lowercase().contains(query)
            })
        })
        .collect();

    let mut start = 0;
    while start < controls.len() {
        let section = controls[start].section.as_str();
        let end = controls[start..]
            .iter()
            .position(|control| control.section != section)
            .map_or(controls.len(), |offset| start + offset);
        ui.group(|ui| {
            ui.set_min_width(ui.available_width());
            egui::CollapsingHeader::new(section)
                .id_salt(("extra-controls", scope as u8, section))
                .default_open(query.is_some() || matches!(section, "Swapper" | "Face Similarity"))
                .show(ui, |ui| {
                    for control in &controls[start..end] {
                        render_control(ui, state, control);
                    }
                });
        });
        ui.add_space(6.0);
        start = end;
    }
}

fn render_control(ui: &mut egui::Ui, state: &mut ExtraGuiState, control: &ControlSpec) {
    let indent = 14.0 * f32::from(control.level.saturating_sub(1));
    let current = scoped_control_state(state, control.scope)
        .get(&control.id)
        .cloned();
    let mut pending = None;

    match (&control.kind, current) {
        (ControlKind::Toggle { default }, Some(ControlValue::Toggle(current))) => {
            let mut value = current;
            ui.horizontal(|ui| {
                ui.add_space(indent);
                let response = ui.checkbox(&mut value, &control.label);
                if !control.help.is_empty() {
                    response.on_hover_text(&control.help);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if reset_button(ui, value != *default) {
                        pending = Some(ControlValue::Toggle(*default));
                    }
                });
            });
            if value != current {
                pending = Some(ControlValue::Toggle(value));
            }
        }
        (
            ControlKind::Slider {
                min,
                max,
                default,
                step,
            },
            Some(ControlValue::Slider(current)),
        ) => {
            let mut value = current;
            ui.horizontal(|ui| {
                ui.add_space(indent);
                let label = ui.label(&control.label);
                if !control.help.is_empty() {
                    label.on_hover_text(&control.help);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if reset_button(ui, value != *default) {
                        pending = Some(ControlValue::Slider(*default));
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.add_space(indent);
                let numeric_width = 62.0;
                let spacing = ui.spacing().item_spacing.x;
                let slider_width = (ui.available_width() - numeric_width - spacing).max(80.0);
                let response = ui.add_sized(
                    [slider_width, ui.spacing().interact_size.y],
                    egui::Slider::new(&mut value, *min..=*max)
                        .step_by(*step)
                        .show_value(false),
                );
                let numeric = ui.add_sized(
                    [numeric_width, ui.spacing().interact_size.y],
                    egui::DragValue::new(&mut value)
                        .range(*min..=*max)
                        .speed(*step)
                        .min_decimals(slider_decimals(*step))
                        .max_decimals(slider_decimals(*step)),
                );
                let mut changed = response.changed() || numeric.changed();
                if scroll_changes_values(state) && response.hovered() {
                    let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        value = (value + scroll.signum() as f64 * *step).clamp(*min, *max);
                        ui.input_mut(|input| input.smooth_scroll_delta = egui::Vec2::ZERO);
                        changed = true;
                    }
                }
                if changed {
                    pending = Some(ControlValue::Slider(value));
                }
            });
        }
        (
            ControlKind::Choice {
                options,
                default,
                source,
            },
            Some(ControlValue::Choice(current)),
        ) => {
            let choices = source
                .and_then(|_| state.dynamic_choices.get(&control.id))
                .filter(|dynamic| !dynamic.is_empty())
                .unwrap_or(options);
            let mut value = current.clone();
            ui.horizontal(|ui| {
                ui.add_space(indent);
                let label = ui.label(&control.label);
                if !control.help.is_empty() {
                    label.on_hover_text(&control.help);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if reset_button(ui, value != *default) {
                        pending = Some(ControlValue::Choice(default.clone()));
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.add_space(indent);
                if choices.is_empty() {
                    ui.add_enabled(false, egui::Button::new("No models found"));
                    return;
                }
                let selected = if value.is_empty() {
                    "Choose…"
                } else {
                    value.as_str()
                };
                let combo = egui::ComboBox::from_id_salt(("extra-choice", &control.id))
                    .selected_text(selected)
                    .width(ui.available_width().max(120.0))
                    .show_ui(ui, |ui| {
                        for option in choices {
                            ui.selectable_value(&mut value, option.clone(), option);
                        }
                    });
                if scroll_changes_values(state) && combo.response.hovered() {
                    let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0
                        && let Some(index) = choices.iter().position(|choice| choice == &value)
                    {
                        let next = if scroll > 0.0 {
                            index.saturating_sub(1)
                        } else {
                            (index + 1).min(choices.len().saturating_sub(1))
                        };
                        value = choices[next].clone();
                        ui.input_mut(|input| input.smooth_scroll_delta = egui::Vec2::ZERO);
                    }
                }
            });
            if value != current {
                pending = Some(ControlValue::Choice(value));
            }
        }
        _ => {}
    }

    if let Some(value) = pending {
        apply_control(state, control, value);
    }
    ui.add_space(3.0);
}

fn reset_button(ui: &mut egui::Ui, enabled: bool) -> bool {
    ui.add_enabled(enabled, egui::Button::new("Reset").small())
        .clicked()
}

fn scroll_changes_values(state: &ExtraGuiState) -> bool {
    matches!(
        state.controls.get("ScrollChangesValuesToggle"),
        Some(ControlValue::Toggle(true))
    )
}

fn slider_decimals(step: f64) -> usize {
    if step >= 1.0 {
        0
    } else if step >= 0.1 {
        1
    } else {
        2
    }
}

fn apply_control(state: &mut ExtraGuiState, control: &ControlSpec, value: ControlValue) {
    let result = if control.scope == ControlScope::Settings {
        state.controls.set(&control.id, value, &state.catalog)
    } else if let Some(target_id) = state.faces.selected_target().map(str::to_owned) {
        state
            .faces
            .target_mut(&target_id)
            .expect("selected target must exist")
            .controls
            .set(&control.id, value, &state.catalog)
    } else {
        state.controls.set(&control.id, value, &state.catalog)
    };
    match result {
        Ok(()) => {
            state.status = format!("Updated {}", control.label);
            state.dirty = true;
        }
        Err(error) => state.status = error.to_string(),
    }
}

fn scoped_control_state(
    state: &ExtraGuiState,
    scope: ControlScope,
) -> &super::controls::ControlState {
    if scope == ControlScope::Settings {
        return &state.controls;
    }
    state
        .faces
        .selected_target()
        .and_then(|id| state.faces.target(id))
        .map_or(&state.controls, |target| &target.controls)
}
