use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::settings::ExecutionProvider;

const BG: egui::Color32 = egui::Color32::from_rgb(0x0d, 0x0d, 0x11);
const PANEL: egui::Color32 = egui::Color32::from_rgb(0x1c, 0x1e, 0x22);
const BORDER: egui::Color32 = egui::Color32::from_rgb(0x36, 0x3a, 0x42);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x34, 0x78, 0xf6);
const TEXT: egui::Color32 = egui::Color32::from_rgb(0xf4, 0xf5, 0xf8);
const TEXT_DIM: egui::Color32 = egui::Color32::from_rgb(0x91, 0x97, 0xa8);

pub(super) fn viewport_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("noperson-realtime-preview")
}

#[derive(Clone)]
pub(super) struct RealtimePreviewSnapshot {
    pub(super) texture: Option<egui::load::SizedTexture>,
    pub(super) fps: f32,
    pub(super) face_count: usize,
    pub(super) provider: ExecutionProvider,
    pub(super) status: String,
}

#[derive(Clone, Default)]
pub(super) struct RealtimePreviewState {
    open: bool,
    close_requested: Arc<AtomicBool>,
}

impl RealtimePreviewState {
    pub(super) fn open(&mut self) {
        self.close_requested.store(false, Ordering::Release);
        self.open = true;
    }

    pub(super) fn close(&mut self, ctx: &egui::Context) {
        let was_open = self.open;
        self.open = false;
        self.close_requested.store(false, Ordering::Release);
        if was_open {
            ctx.send_viewport_cmd_to(viewport_id(), egui::ViewportCommand::Close);
        }
    }

    pub(super) fn mark_close_requested(&self) {
        self.close_requested.store(true, Ordering::Release);
    }

    pub(super) fn apply_close_request(&mut self) -> bool {
        if self.close_requested.swap(false, Ordering::AcqRel) {
            self.open = false;
            true
        } else {
            false
        }
    }

    pub(super) fn is_open(&self) -> bool {
        self.open
    }
}

pub(super) fn fit_image_size(image: egui::Vec2, available: egui::Vec2) -> egui::Vec2 {
    if image.x <= 0.0 || image.y <= 0.0 || available.x <= 0.0 || available.y <= 0.0 {
        return egui::Vec2::ZERO;
    }
    image * (available.x / image.x).min(available.y / image.y)
}

fn provider_label(provider: ExecutionProvider) -> &'static str {
    match provider {
        ExecutionProvider::Cuda => "Native CUDA",
        ExecutionProvider::TensorRT => "TensorRT",
    }
}

pub(super) fn render_content(ui: &mut egui::Ui, snapshot: &RealtimePreviewSnapshot) {
    ui.painter().rect_filled(ui.max_rect(), 0.0, BG);
    egui::Frame::NONE
        .fill(BG)
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            let header = egui::Frame::NONE
                .fill(PANEL)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        egui::Frame::NONE
                            .fill(ACCENT)
                            .corner_radius(5.0)
                            .inner_margin(egui::Margin::symmetric(9, 4))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new("LIVE")
                                        .color(egui::Color32::WHITE)
                                        .strong()
                                        .size(12.0),
                                );
                            });
                        ui.separator();
                        ui.label(
                            egui::RichText::new(provider_label(snapshot.provider))
                                .color(TEXT_DIM)
                                .strong(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(format!("{} faces", snapshot.face_count))
                                    .color(TEXT_DIM)
                                    .strong(),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:.1} FPS", snapshot.fps))
                                    .color(ACCENT)
                                    .strong()
                                    .size(22.0),
                            );
                        });
                    });
                });

            let stage_rect = egui::Rect::from_min_max(
                egui::pos2(ui.max_rect().left(), header.response.rect.bottom() + 10.0),
                ui.max_rect().right_bottom(),
            );
            ui.allocate_rect(stage_rect, egui::Sense::hover());
            if let Some(texture) = &snapshot.texture {
                let image_rect = egui::Rect::from_center_size(
                    stage_rect.center(),
                    fit_image_size(texture.size, stage_rect.size()),
                );
                let corner_radius = egui::CornerRadius::same(12);
                ui.painter().add(
                    egui::epaint::Shadow {
                        offset: [0, 5],
                        blur: 18,
                        spread: 1,
                        color: egui::Color32::from_black_alpha(150),
                    }
                    .as_shape(image_rect, corner_radius),
                );
                egui::Image::new(*texture)
                    .fit_to_exact_size(image_rect.size())
                    .corner_radius(corner_radius)
                    .alt_text("Live swapped output")
                    .paint_at(ui, image_rect);
                ui.painter().rect_stroke(
                    image_rect,
                    corner_radius,
                    egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)),
                    egui::StrokeKind::Inside,
                );
                render_video_overlay(ui, image_rect, snapshot);
            } else {
                let spinner_size = egui::vec2(24.0, 24.0);
                let spinner_rect = egui::Rect::from_center_size(
                    stage_rect.center() - egui::vec2(0.0, 18.0),
                    spinner_size,
                );
                ui.put(spinner_rect, egui::Spinner::new().size(22.0));
                let status_rect = egui::Rect::from_center_size(
                    stage_rect.center() + egui::vec2(0.0, 18.0),
                    egui::vec2(stage_rect.width().min(420.0), 24.0),
                );
                ui.put(
                    status_rect,
                    egui::Label::new(egui::RichText::new(&snapshot.status).color(TEXT).size(14.0))
                        .halign(egui::Align::Center),
                );
            }
        });
}

fn render_video_overlay(
    ui: &mut egui::Ui,
    image_rect: egui::Rect,
    snapshot: &RealtimePreviewSnapshot,
) {
    let face_suffix = if snapshot.face_count == 1 {
        "face"
    } else {
        "faces"
    };
    let text = format!(
        "● {:.1} FPS · {} · {} {face_suffix}",
        snapshot.fps,
        provider_label(snapshot.provider),
        snapshot.face_count
    );
    let overlay_size = egui::vec2(260.0_f32.min(image_rect.width() - 20.0), 32.0);
    let overlay_rect = egui::Rect::from_min_size(
        egui::pos2(
            image_rect.right() - overlay_size.x - 10.0,
            image_rect.top() + 10.0,
        ),
        overlay_size,
    );
    ui.painter().rect(
        overlay_rect,
        8.0,
        egui::Color32::from_black_alpha(190),
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(28)),
        egui::StrokeKind::Inside,
    );
    ui.put(
        overlay_rect.shrink2(egui::vec2(10.0, 5.0)),
        egui::Label::new(
            egui::RichText::new(text)
                .color(egui::Color32::from_rgb(0x9e, 0xeb, 0xc5))
                .strong()
                .size(11.0),
        )
        .halign(egui::Align::Center),
    );
}

pub(super) fn show(
    ctx: &egui::Context,
    state: &RealtimePreviewState,
    snapshot: RealtimePreviewSnapshot,
) {
    if !state.is_open() {
        return;
    }

    let close_state = state.clone();
    ctx.show_viewport_deferred(
        viewport_id(),
        egui::ViewportBuilder::default()
            .with_title("noperson — Live preview")
            .with_inner_size([960.0, 600.0])
            .with_min_inner_size([480.0, 320.0])
            .with_resizable(true),
        move |ui, _class| {
            if ui.ctx().input(|input| input.viewport().close_requested()) {
                close_state.mark_close_requested();
            }
            render_content(ui, &snapshot);
        },
    );
    ctx.request_repaint_of(viewport_id());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::settings::ExecutionProvider;
    use egui_kittest::{Harness, kittest::Queryable as _};

    fn assert_vec2_close(actual: egui::Vec2, expected: egui::Vec2) {
        assert!(
            (actual.x - expected.x).abs() < 0.001,
            "x: expected {}, got {}",
            expected.x,
            actual.x
        );
        assert!(
            (actual.y - expected.y).abs() < 0.001,
            "y: expected {}, got {}",
            expected.y,
            actual.y
        );
    }

    #[test]
    fn live_video_surface_uses_the_product_canvas() {
        assert_eq!(BG, egui::Color32::from_rgb(0x0d, 0x0d, 0x11));
    }

    #[test]
    fn image_fit_preserves_aspect_ratio_in_wide_and_tall_windows() {
        assert_vec2_close(
            fit_image_size(egui::vec2(1920.0, 1080.0), egui::vec2(800.0, 400.0)),
            egui::vec2(711.1111, 400.0),
        );
        assert_vec2_close(
            fit_image_size(egui::vec2(1920.0, 1080.0), egui::vec2(400.0, 800.0)),
            egui::vec2(400.0, 225.0),
        );
    }

    #[test]
    fn warmup_content_shows_live_metrics_and_status() {
        let snapshot = RealtimePreviewSnapshot {
            texture: None,
            fps: 0.0,
            face_count: 0,
            provider: ExecutionProvider::Cuda,
            status: "Opening webcam".to_owned(),
        };
        let harness = Harness::builder()
            .with_size(egui::vec2(960.0, 600.0))
            .build_ui(|ui| render_content(ui, &snapshot));

        for label in [
            "LIVE",
            "Native CUDA",
            "0.0 FPS",
            "0 faces",
            "Opening webcam",
        ] {
            assert!(
                harness.query_by_label(label).is_some(),
                "missing preview label: {label}"
            );
        }
    }

    #[test]
    #[ignore = "local visual baseline is stored in ignored tests/snapshots"]
    fn warmup_visual_snapshot() {
        let snapshot = RealtimePreviewSnapshot {
            texture: None,
            fps: 0.0,
            face_count: 0,
            provider: ExecutionProvider::Cuda,
            status: "Opening webcam".to_owned(),
        };
        let mut harness = Harness::builder()
            .with_size(egui::vec2(960.0, 600.0))
            .wgpu()
            .build_ui(move |ui| render_content(ui, &snapshot));

        harness.snapshot("realtime_preview_warmup");
    }

    #[test]
    #[ignore = "local visual baseline is stored in ignored tests/snapshots"]
    fn frame_visual_snapshot() {
        let image = image::open("assets/photos/face.jpg")
            .expect("face.jpg test fixture")
            .thumbnail(640, 360)
            .to_rgb8();
        let color_image = egui::ColorImage::from_rgb(
            [image.width() as usize, image.height() as usize],
            image.as_raw(),
        );
        let mut harness = Harness::builder()
            .with_size(egui::vec2(960.0, 600.0))
            .wgpu()
            .build_ui(move |ui| {
                let texture = ui.ctx().load_texture(
                    "realtime-preview-test-frame",
                    color_image.clone(),
                    egui::TextureOptions::LINEAR,
                );
                render_content(
                    ui,
                    &RealtimePreviewSnapshot {
                        texture: Some(egui::load::SizedTexture::from_handle(&texture)),
                        fps: 27.4,
                        face_count: 1,
                        provider: ExecutionProvider::TensorRT,
                        status: "Running".to_owned(),
                    },
                );
            });

        harness.snapshot("realtime_preview_frame");
    }

    #[test]
    fn live_frame_renders_a_single_glass_status_overlay() {
        let color_image = egui::ColorImage::from_rgb([2, 2], &[32; 12]);
        let harness = Harness::builder()
            .with_size(egui::vec2(960.0, 600.0))
            .build_ui(move |ui| {
                let texture = ui.ctx().load_texture(
                    "glass-overlay-test-frame",
                    color_image.clone(),
                    egui::TextureOptions::LINEAR,
                );
                render_content(
                    ui,
                    &RealtimePreviewSnapshot {
                        texture: Some(egui::load::SizedTexture::from_handle(&texture)),
                        fps: 27.4,
                        face_count: 1,
                        provider: ExecutionProvider::TensorRT,
                        status: "Running".to_owned(),
                    },
                );
            });

        assert!(
            harness
                .query_by_label("● 27.4 FPS · TensorRT · 1 face")
                .is_some(),
            "live metrics must be rendered over the video as one glass pill"
        );
    }
}
