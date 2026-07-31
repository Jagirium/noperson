use std::collections::HashSet;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Matrix {
    schema: u32,
    domains: Vec<Domain>,
    excluded: Vec<Excluded>,
}

#[derive(Debug, Deserialize)]
struct Domain {
    name: String,
    items: Vec<Item>,
}

#[derive(Debug, Deserialize)]
struct Item {
    python: String,
    rust: Option<String>,
    status: String,
    evidence: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Excluded {
    capability: String,
    reason: String,
}

#[test]
fn parity_matrix_is_complete_and_unambiguous() {
    let raw = include_str!("fixtures/crosswap-backend-matrix.json");
    let matrix: Matrix = serde_json::from_str(raw).expect("valid parity matrix");
    assert_eq!(matrix.schema, 1);

    let required_domains = [
        "model-lifecycle",
        "face-detection",
        "face-landmarks",
        "recognition-and-swapping",
        "masks",
        "face-restoration",
        "frame-enhancement",
        "dfm",
        "media-pipeline",
    ];
    let actual_domains: HashSet<_> = matrix
        .domains
        .iter()
        .map(|domain| domain.name.as_str())
        .collect();
    for required in required_domains {
        assert!(
            actual_domains.contains(required),
            "missing domain {required}"
        );
    }

    let mut python_symbols = HashSet::new();
    for domain in &matrix.domains {
        assert!(!domain.items.is_empty(), "empty domain {}", domain.name);
        for item in &domain.items {
            assert!(
                python_symbols.insert(item.python.as_str()),
                "duplicate {}",
                item.python
            );
            assert!(matches!(
                item.status.as_str(),
                "missing" | "partial" | "parity" | "verified"
            ));
            if matches!(item.status.as_str(), "parity" | "verified") {
                assert!(
                    item.rust.is_some(),
                    "resolved item {} has no Rust symbol",
                    item.python
                );
                assert!(
                    !item.evidence.is_empty(),
                    "resolved item {} has no evidence",
                    item.python
                );
            }
        }
    }

    let exclusions: HashSet<_> = matrix
        .excluded
        .iter()
        .map(|item| item.capability.as_str())
        .collect();
    assert!(exclusions.contains("background-removal"));
    assert!(exclusions.contains("GPEN-1024"));
    assert!(exclusions.contains("GPEN-2048"));
    assert!(
        matrix
            .excluded
            .iter()
            .all(|item| !item.reason.trim().is_empty())
    );
}
