//! Static model registry: release assets, BLAKE3 identities, and mirrors.
//!
//! Port of crosswap/app/processors/models_data.py

pub use crate::artifacts::{ArtifactEntry as ModelEntry, ArtifactMirror as ModelMirror};

macro_rules! model {
    ($name:literal, $filename:literal, $size:literal, $blake3:literal) => {
        ModelEntry {
            name: $name,
            filename: $filename,
            size: $size,
            blake3: $blake3,
            mirrors: [
                ModelMirror {
                    name: "github",
                    url: concat!(
                        "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/",
                        $filename
                    ),
                },
                ModelMirror {
                    name: "hugging-face",
                    url: concat!(
                        "https://huggingface.co/Jagirium/noperson-models/resolve/main/",
                        $filename
                    ),
                },
            ],
        }
    };
}

pub const MODELS: &[ModelEntry] = &[
    model!(
        "Inswapper128",
        "inswapper_128.fp16.onnx",
        277_683_908,
        "65d91b580afffee6c0358c0fb48f9743a5502cbf8fff997e8051f76e07ae7bd0"
    ),
    model!(
        "InswapperEMap",
        "emap.bin",
        1_048_576,
        "a54d56087b426005a6d77c9df3dcddfdb81ebc610d6af2235181ef2f97bae4b7"
    ),
    model!(
        "Inswapper128ArcFace",
        "w600k_r50.onnx",
        174_383_860,
        "3899d93a9ca878fa782fd474078fcb999b94c78a09e93bb079d9b0904f7f0a8b"
    ),
    model!(
        "RetinaFace",
        "det_10g.onnx",
        16_926_583,
        "e75572cd5e57d30163b7375f475dca0d4b7dfb799788bd670c7065ea045494de"
    ),
    model!(
        "SCRFD2.5g",
        "scrfd_2.5g_bnkps.onnx",
        3_290_207,
        "203ba550b5c1979d0fa918fe7a5a5dd1fe2adf94d49d886225cfb59f03a52292"
    ),
    model!(
        "YoloFace8n",
        "yoloface_8n.onnx",
        12_639_351,
        "d9e2c8b06b021310fa58059e35af1db46940e183f352a94fb51a786ccb094a26"
    ),
    model!(
        "FaceLandmark5",
        "res50.onnx",
        109_105_339,
        "b1d07cf46163e82d7f2517e304f14f36e10a88970cd4c8e2976b69d78a620e3b"
    ),
    model!(
        "FaceLandmark68",
        "2dfan4.onnx",
        97_904_402,
        "bfe070138976aa210066058e732b244789f099434027432d68c7e5f2f437832c"
    ),
    model!(
        "FaceLandmark3d68",
        "1k3d68.onnx",
        143_507_290,
        "a44d506f23592f687cae2eaccc407fe09ebb646c1bc2496ac7642476db1a4ec2"
    ),
    model!(
        "FaceLandmark98",
        "peppapig_teacher_Nx3x256x256.onnx",
        43_701_237,
        "53aecd61bb2d596ef889bb91de354e444e073341a77934670d086310773be9e8"
    ),
    model!(
        "FaceLandmark106",
        "2d106det.onnx",
        5_030_888,
        "29089d5c735695a4d60ebcb81075dc800ac7682386029d328291b64bcc5ac7ba"
    ),
    model!(
        "FaceLandmark203",
        "landmark.onnx",
        114_662_596,
        "b45505c4e5f23c8b1e101500d63b61a0e46cc74a975945c815bb1eab6d8b09bd"
    ),
    model!(
        "FaceLandmark478",
        "face_landmarks_detector_Nx3x256x256.onnx",
        4_955_447,
        "78442afd94239bca732db32b8f54328c916788caf3a66b157d025f54d1d23aee"
    ),
    model!(
        "FaceBlendShapes",
        "face_blendshapes_Nx146x2.onnx",
        1_862_967,
        "08e963b266ffecfdd1ee864b64a3f87ebec8fe8c9c7f303075a429cca39dcff4"
    ),
    model!(
        "GPENBFR256",
        "GPEN-BFR-256.onnx",
        75_642_939,
        "1f02932360eb61b7d871bf59055e5b3bf94d8282f114522e158b8617adda1637"
    ),
    model!(
        "GPENBFR512",
        "GPEN-BFR-512.onnx",
        284_166_354,
        "bf5a90dc4c85099e6b405895466e6bb69ce96fed56db58d057c34e5ce4a7e56f"
    ),
    model!(
        "RealEsrganx2Plus",
        "RealESRGAN_x2plus.fp16.onnx",
        33_585_997,
        "d8887f141735b658d11738efbac4baf9a28ce8d550025ce5dddbcd35df3818df"
    ),
    model!(
        "RealEsrganx4Plus",
        "RealESRGAN_x4plus.fp16.onnx",
        33_606_180,
        "ba140362621bf299dedf15b5c630a28c568dc7a28cea96e3122e4d45f726224f"
    ),
    model!(
        "RealEsrx4v3",
        "realesr-general-x4v3.onnx",
        4_871_181,
        "743285537a5749abfc4dd0e3d864d8991be1d0089ada13c14ad0dc765ae04acf"
    ),
    model!(
        "BSRGANx2",
        "BSRGANx2.fp16.onnx",
        33_531_571,
        "e2a785a114f879f343154d8eb05090d2ea1a84efc978e0453c76b25b9c47aaca"
    ),
    model!(
        "BSRGANx4",
        "BSRGANx4.fp16.onnx",
        33_606_180,
        "48fbf7861b04e270b5a28fd14ac558c2da0b021c216d3420308411cdf6db83e2"
    ),
    model!(
        "UltraSharpx4",
        "4x-UltraSharp.fp16.onnx",
        33_650_272,
        "896bd3d1474665a33f3ec9e8eadf1a2ec66eb84f9b7a2a5e8ce3eda969e33cf1"
    ),
    model!(
        "UltraMixx4",
        "4x-UltraMix_Smooth.fp16.onnx",
        33_606_180,
        "65111906b7e08c86294062e7670d8f0e77ef33c468e3ecb705bd1ea8274f8c9b"
    ),
    model!(
        "Occluder",
        "occluder.onnx",
        57_308_418,
        "7d60d9f841da204f4b97cb15c7b039d114f6927e1a3ae1f0cd7c8375a8e11c00"
    ),
    model!(
        "XSeg",
        "XSeg_model.onnx",
        70_324_682,
        "23296e0625965f2a0b19a8c74d96dc15e302c8c8323d883e14dd1c566b44c204"
    ),
    model!(
        "FaceParser",
        "faceparser_resnet34.onnx",
        93_637_555,
        "dd480fe0e7bd9bfd2270aa141fcb55afc9e00c78cce4b9b9095d563d4ac9ada8"
    ),
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
    use super::{MODEL_IO, MODELS, find_model};
    use std::collections::HashSet;

    #[test]
    fn model_downloads_have_two_first_party_blake3_mirrors() {
        let github = "https://github.com/Jagirium/noperson/releases/download/models-v0.1.0/";
        let hugging_face = "https://huggingface.co/Jagirium/noperson-models/resolve/main/";
        let mut urls = HashSet::new();

        assert_eq!(MODELS.len(), 26);
        for model in MODELS {
            assert!(
                model.filename.ends_with(".onnx") || model.filename == "emap.bin",
                "unexpected release asset registered: {}",
                model.filename
            );
            assert!(model.size > 0, "missing byte size for {}", model.name);
            assert_eq!(model.blake3.len(), 64, "invalid BLAKE3 for {}", model.name);
            assert_ne!(
                model.blake3, "0000000000000000000000000000000000000000000000000000000000000000",
                "missing release BLAKE3 for {}",
                model.name
            );
            assert!(
                model
                    .blake3
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "BLAKE3 must be lowercase hexadecimal for {}",
                model.name
            );
            assert_eq!(model.mirrors.len(), 2, "{} needs two mirrors", model.name);
            assert_eq!(model.mirrors[0].url, format!("{github}{}", model.filename));
            assert_eq!(
                model.mirrors[1].url,
                format!("{hugging_face}{}", model.filename)
            );
            for mirror in model.mirrors {
                assert!(urls.insert(mirror.url), "duplicate URL for {}", model.name);
            }
        }
    }

    #[test]
    fn runtime_registry_excludes_oversized_gpen_variants() {
        assert!(find_model("GPENBFR1024").is_none());
        assert!(find_model("GPENBFR2048").is_none());
        assert!(
            MODEL_IO
                .iter()
                .all(|model| model.name != "GPENBFR1024" && model.name != "GPENBFR2048")
        );
    }
}
