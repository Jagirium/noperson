//! Static model registry: names, ONNX paths, SHA256, download URLs.
//!
//! Port of crosswap/app/processors/models_data.py

pub struct ModelEntry {
    pub name: &'static str,
    pub filename: &'static str,
    pub sha256: Option<&'static str>,
    pub url: &'static str,
}

pub const MODELS: &[ModelEntry] = &[
    ModelEntry {
        name: "Inswapper128",
        filename: "inswapper_128.fp16.onnx",
        sha256: Some("f29a902862df018264ad4fd0c25387acd0581e168a9baa0372d71c465b65bf27"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/inswapper_128.fp16.onnx",
    },
    ModelEntry {
        name: "Inswapper128ArcFace",
        filename: "w600k_r50.onnx",
        sha256: Some("0ddde02d672b5063bb79641844d4e938c9e1f1a66607b4e2436ef15036fe7c9a"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/w600k_r50.onnx",
    },
    ModelEntry {
        name: "RetinaFace",
        filename: "det_10g.onnx",
        sha256: Some("40c91393416f47d4af83c21cd8fce5f4b025ed50e6bd2261b00d91587c9ef0d3"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/det_10g.onnx",
    },
    ModelEntry {
        name: "SCRFD2.5g",
        filename: "scrfd_2.5g_bnkps.onnx",
        sha256: Some("bc24bb349491481c3ca793cf89306723162c280cb284c5a5e49df3760bf5c2ce"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/scrfd_2.5g_bnkps.onnx",
    },
    ModelEntry {
        name: "YoloFace8n",
        filename: "yoloface_8n.onnx",
        sha256: Some("84d5bb985b0ea75fc851d7454483897b1494c71c211759b4fec3a22ac196d206"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/yoloface_8n.onnx",
    },
    ModelEntry {
        name: "FaceLandmark5",
        filename: "res50.onnx",
        sha256: Some("4b6b1f0bb9fc00f2901e332cdbe7b311653aee67b2fc19433c6b3818f7059a3b"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/res50.onnx",
    },
    ModelEntry {
        name: "FaceLandmark68",
        filename: "2dfan4.onnx",
        sha256: Some("1ceedb108439c7d7b3f92cfa2b25bdc69a1f5f6c8b41da228cb283ca98d4181d"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/2dfan4.onnx",
    },
    ModelEntry {
        name: "FaceLandmark3d68",
        filename: "1k3d68.onnx",
        sha256: Some("37f8bab824d59ecada10f5540bb7a061596728ebe0860f95488b8679fc746d91"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/1k3d68.onnx",
    },
    ModelEntry {
        name: "FaceLandmark98",
        filename: "peppapig_teacher_Nx3x256x256.onnx",
        sha256: Some("d4aa6dbd0081763a6eef04bf51484175b6a133ed12999bdc83b681a03f3f87d2"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/peppapig_teacher_Nx3x256x256.onnx",
    },
    ModelEntry {
        name: "FaceLandmark106",
        filename: "2d106det.onnx",
        sha256: Some("f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/2d106det.onnx",
    },
    ModelEntry {
        name: "FaceLandmark203",
        filename: "landmark.onnx",
        sha256: Some("e10c5ab774fc9016be0f59ec2979ec93930e36d40bab54283f06f8eff14cfd6e"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/landmark.onnx",
    },
    ModelEntry {
        name: "FaceLandmark478",
        filename: "face_landmarks_detector_Nx3x256x256.onnx",
        sha256: Some("6d7932bdefc38871f57dd915b8c723d855e599f29cf4cdf19616fb35d0ed572e"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/face_landmarks_detector_Nx3x256x256.onnx",
    },
    ModelEntry {
        name: "FaceBlendShapes",
        filename: "face_blendshapes_Nx146x2.onnx",
        sha256: Some("79065a18016da3b95f71247ff9ade3fe09b9124903a26a1af85af6d9e2a4faf3"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/face_blendshapes_Nx146x2.onnx",
    },
    ModelEntry {
        name: "GPENBFR256",
        filename: "GPEN-BFR-256.onnx",
        sha256: Some("65d8c6ba3cea12a7fef7ab2bb1c8ce27def5ce61958024fcf2a72abbffaaa442"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/GPEN-BFR-256.onnx",
    },
    ModelEntry {
        name: "GPENBFR512",
        filename: "GPEN-BFR-512.onnx",
        sha256: Some("e4394805876f114a448c405bf66c81ca2cad1aebeead0b91aa9313fba5ffd122"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/GPEN-BFR-512.onnx",
    },
    ModelEntry {
        name: "GPENBFR1024",
        filename: "GPEN-BFR-1024.onnx",
        sha256: Some("2a579f36202f451a5e1fcf69e1ae8ce25d156a08cd5e43d484c44da16cd6c6fe"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/GPEN-BFR-1024.onnx",
    },
    ModelEntry {
        name: "RealEsrganx2Plus",
        filename: "RealESRGAN_x2plus.fp16.onnx",
        sha256: Some("80f8b0f9cfaa7b3e972495bd21291f027aa60bf66af9c38d58c52cdf086b0a59"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/RealESRGAN_x2plus.fp16.onnx",
    },
    ModelEntry {
        name: "RealEsrganx4Plus",
        filename: "RealESRGAN_x4plus.fp16.onnx",
        sha256: Some("0a06c68f463a14bf5563b78d77d61ba4394024e148383c4308d6d3783eac2dc5"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/RealESRGAN_x4plus.fp16.onnx",
    },
    ModelEntry {
        name: "RealEsrx4v3",
        filename: "realesr-general-x4v3.onnx",
        sha256: Some("09b757accd747d7e423c1d352b3e8f23e77cc5742d04bae958d4eb8082b76fa4"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/realesr-general-x4v3.onnx",
    },
    ModelEntry {
        name: "BSRGANx2",
        filename: "BSRGANx2.fp16.onnx",
        sha256: Some("ba3a43613f5d2434c853201411b87e75c25ccb5b5918f38af504e4cf3bd4df9a"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/BSRGANx2.fp16.onnx",
    },
    ModelEntry {
        name: "BSRGANx4",
        filename: "BSRGANx4.fp16.onnx",
        sha256: Some("e1467fbe60d2846919480f55a12ddbd5c516e343685bcdeac50ddcfa1dde2f46"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/BSRGANx4.fp16.onnx",
    },
    ModelEntry {
        name: "UltraSharpx4",
        filename: "4x-UltraSharp.fp16.onnx",
        sha256: Some("50ee59f866246ebc5dfe0a08aa62de65fc8179e3abe3de8b6438e545bfdbebaf"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/4x-UltraSharp.fp16.onnx",
    },
    ModelEntry {
        name: "UltraMixx4",
        filename: "4x-UltraMix_Smooth.fp16.onnx",
        sha256: Some("3b96d63c239121b1ad5992e42a2089d6b4e1185c493c6440adfeafc0a20591eb"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/4x-UltraMix_Smooth.fp16.onnx",
    },
    ModelEntry {
        name: "Occluder",
        filename: "occluder.onnx",
        sha256: Some("79f5c2edf10b83458693d122dd51488b210fb80c059c5d56347a047710d44a78"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/occluder.onnx",
    },
    ModelEntry {
        name: "XSeg",
        filename: "XSeg_model.onnx",
        sha256: Some("4381395dcbec1eef469fa71cfb381f00ac8aadc3e5decb4c29c36b6eb1f38ad9"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/XSeg_model.onnx",
    },
    ModelEntry {
        name: "FaceParser",
        filename: "faceparser_resnet34.onnx",
        sha256: Some("688eb7365229a77e3c5ccb32c018e57340b3b7947f18ed8adf98f143719d9b7e"),
        url: "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/faceparser_resnet34.onnx",
    },
];

/// Find a model entry by name.
pub fn find_model(name: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().find(|m| m.name == name)
}

/// ArcFace model mapping: swapper → arcface model.
pub const ARCFACE_MAPPING: &[(&str, &str)] = &[
    ("Inswapper128", "Inswapper128ArcFace"),
    ("DeepFaceLive (DFM)", "Inswapper128ArcFace"),
];

/// Detection model mapping: UI name → model name.
pub const DETECTION_MODEL_MAPPING: &[(&str, &str)] = &[
    ("RetinaFace", "RetinaFace"),
    ("SCRFD", "SCRFD2.5g"),
    ("Yolov8", "YoloFace8n"),
];

/// Landmark model mapping: point count → model name.
pub const LANDMARK_MODEL_MAPPING: &[(&str, &str)] = &[
    ("5", "FaceLandmark5"),
    ("68", "FaceLandmark68"),
    ("3d68", "FaceLandmark3d68"),
    ("98", "FaceLandmark98"),
    ("106", "FaceLandmark106"),
    ("203", "FaceLandmark203"),
    ("478", "FaceLandmark478"),
];

/// ONNX I/O specification for each model.
pub struct ModelIO {
    pub name: &'static str,
    pub input_names: &'static [&'static str],
    pub output_names: &'static [&'static str],
}

pub const MODEL_IO: &[ModelIO] = &[
    ModelIO {
        name: "YoloFace8n",
        input_names: &["images"],
        output_names: &["output0"],
    },
    ModelIO {
        name: "RetinaFace",
        input_names: &["input.1"],
        output_names: &[
            "448", "451", "454", "471", "474", "477", "494", "497", "500",
        ],
    },
    ModelIO {
        name: "Inswapper128",
        input_names: &["target", "source"],
        output_names: &["output"],
    },
    ModelIO {
        name: "Occluder",
        input_names: &["img"],
        output_names: &["output"],
    },
    ModelIO {
        name: "XSeg",
        input_names: &["in_face:0"],
        output_names: &["out_mask:0"],
    },
    ModelIO {
        name: "FaceParser",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "GPENBFR256",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "GPENBFR512",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "GPENBFR1024",
        input_names: &["input"],
        output_names: &["output"],
    },
    // Frame enhancers — all use input/output
    ModelIO {
        name: "RealEsrganx2Plus",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "RealEsrganx4Plus",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "RealEsrx4v3",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "BSRGANx2",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "BSRGANx4",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "UltraSharpx4",
        input_names: &["input"],
        output_names: &["output"],
    },
    ModelIO {
        name: "UltraMixx4",
        input_names: &["input"],
        output_names: &["output"],
    },
    // Landmarks — input names vary, discovered at runtime
    ModelIO {
        name: "FaceLandmark5",
        input_names: &["input"],
        output_names: &["conf", "landmarks"],
    },
    ModelIO {
        name: "FaceLandmark68",
        input_names: &["input"],
        output_names: &["landmarks_xyscore", "heatmaps"],
    },
    ModelIO {
        name: "FaceLandmark478",
        input_names: &["input_12"],
        output_names: &["Identity", "Identity_1", "Identity_2"],
    },
];

#[cfg(test)]
mod tests {
    use super::MODELS;
    use std::collections::HashSet;

    #[test]
    fn model_downloads_are_first_party_release_assets() {
        let prefix = "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/";
        let mut urls = HashSet::new();

        assert_eq!(MODELS.len(), 26);
        for model in MODELS {
            assert!(
                model.filename.ends_with(".onnx"),
                "non-ONNX release asset registered: {}",
                model.filename
            );
            assert_eq!(model.url, format!("{prefix}{}", model.filename));
            assert!(urls.insert(model.url), "duplicate URL for {}", model.name);
        }
    }
}
