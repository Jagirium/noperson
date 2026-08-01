//! Immutable downloadable artifact identities shared by models and runtimes.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactMirror {
    pub name: &'static str,
    pub url: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactEntry {
    pub name: &'static str,
    pub filename: &'static str,
    pub size: u64,
    pub blake3: &'static str,
    pub mirrors: [ArtifactMirror; 2],
}
