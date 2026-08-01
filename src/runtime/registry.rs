use crate::artifacts::{ArtifactEntry, ArtifactMirror};

use super::TensorRtShard;

pub const GENERATION: &str = "cuda12.8-cudnn9.20-trt10.16-v1";

macro_rules! runtime_artifact {
    ($name:literal, $filename:literal, $size:literal, $blake3:literal) => {
        ArtifactEntry {
            name: $name,
            filename: $filename,
            size: $size,
            blake3: $blake3,
            mirrors: [
                ArtifactMirror {
                    name: "github",
                    url: concat!(
                        "https://github.com/Jagirium/noperson/releases/download/runtime-v0.1.0/",
                        $filename
                    ),
                },
                ArtifactMirror {
                    name: "hugging-face",
                    url: concat!(
                        "https://huggingface.co/Jagirium/noperson-runtime/resolve/main/",
                        $filename
                    ),
                },
            ],
        }
    };
}

pub const BASE: ArtifactEntry = runtime_artifact!(
    "RuntimeBase",
    "noperson-runtime-base-linux-x86_64-v1.tar.zst",
    1_539_077_722,
    "540fb00771788189f97d3b246e5052b013b7eba4881a4fe1b7ca97bbc967d10a"
);
pub const TRT_BASE: ArtifactEntry = runtime_artifact!(
    "TensorRTBase",
    "noperson-runtime-trt-base-linux-x86_64-v1.tar.zst",
    278_172_609,
    "28b05787345cd15050f8a9aecaf6cb6fcf81073f4820ca26535e88ef2fc7988a"
);
pub const TRT_SM75: ArtifactEntry = runtime_artifact!(
    "TensorRTSm75",
    "noperson-runtime-trt-sm75-linux-x86_64-v1.tar.zst",
    76_632_635,
    "12ca1ade33ccf13198362ed94fe0b5f79ef10e8c193deb6d5bced1fa762d5a50"
);
pub const TRT_SM80: ArtifactEntry = runtime_artifact!(
    "TensorRTSm80",
    "noperson-runtime-trt-sm80-linux-x86_64-v1.tar.zst",
    139_121_974,
    "0394e880ba9a18aedb52b37997680ab84f61edec29ea26e6718d1170002f2780"
);
pub const TRT_SM86: ArtifactEntry = runtime_artifact!(
    "TensorRTSm86",
    "noperson-runtime-trt-sm86-linux-x86_64-v1.tar.zst",
    130_000_202,
    "c4643e645fbe870e7f9db0326c6ca6d4a89e91b7d877a3c22195bd59aca252e5"
);
pub const TRT_SM89: ArtifactEntry = runtime_artifact!(
    "TensorRTSm89",
    "noperson-runtime-trt-sm89-linux-x86_64-v1.tar.zst",
    136_596_398,
    "2c3df470a05a4b077a59674f172c33224bafe11cb568a17aa53ea32dd829324c"
);
pub const TRT_SM90: ArtifactEntry = runtime_artifact!(
    "TensorRTSm90",
    "noperson-runtime-trt-sm90-linux-x86_64-v1.tar.zst",
    420_401_845,
    "1000de3bef2f1abaa0b9b222b970cb8cc1eec8ac94f908fcbb13133fbe562dc0"
);
pub const TRT_SM100: ArtifactEntry = runtime_artifact!(
    "TensorRTSm100",
    "noperson-runtime-trt-sm100-linux-x86_64-v1.tar.zst",
    242_329_853,
    "e7f076c61e71139216d914f447e10f292a15abc7b67eca51b73926d42c11c935"
);
pub const TRT_SM120: ArtifactEntry = runtime_artifact!(
    "TensorRTSm120",
    "noperson-runtime-trt-sm120-linux-x86_64-v1.tar.zst",
    216_910_455,
    "5eecdb14a4ddfb248ef8aabcdafafe24034173c4e7cfa084d2a52e0f56d55df4"
);
pub const TRT_PTX: ArtifactEntry = runtime_artifact!(
    "TensorRTPtx",
    "noperson-runtime-trt-ptx-linux-x86_64-v1.tar.zst",
    245_340_197,
    "0a4122430b11c7fb00d06c6bf12db0aad286bb87c5b309527789af484d91c64b"
);

pub fn artifacts_for(shard: TensorRtShard) -> [&'static ArtifactEntry; 3] {
    let shard = match shard {
        TensorRtShard::Sm75 => &TRT_SM75,
        TensorRtShard::Sm80 => &TRT_SM80,
        TensorRtShard::Sm86 => &TRT_SM86,
        TensorRtShard::Sm89 => &TRT_SM89,
        TensorRtShard::Sm90 => &TRT_SM90,
        TensorRtShard::Sm100 => &TRT_SM100,
        TensorRtShard::Sm120 => &TRT_SM120,
        TensorRtShard::Ptx => &TRT_PTX,
    };
    [&BASE, &TRT_BASE, shard]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn every_runtime_archive_has_two_immutable_blake3_mirrors() {
        let mut filenames = HashSet::new();
        for shard in [
            TensorRtShard::Sm75,
            TensorRtShard::Sm80,
            TensorRtShard::Sm86,
            TensorRtShard::Sm89,
            TensorRtShard::Sm90,
            TensorRtShard::Sm100,
            TensorRtShard::Sm120,
            TensorRtShard::Ptx,
        ] {
            for artifact in artifacts_for(shard) {
                assert!(artifact.size > 0);
                assert_eq!(artifact.blake3.len(), 64);
                assert!(artifact.filename.ends_with(".tar.zst"));
                assert!(artifact.mirrors[0].url.contains("/runtime-v0.1.0/"));
                assert!(artifact.mirrors[1].url.contains("/noperson-runtime/"));
                filenames.insert(artifact.filename);
            }
        }
        assert_eq!(filenames.len(), 10);
    }
}
