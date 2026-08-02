use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ControlScope {
    Settings,
    Common,
    Swapper,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChoiceSource {
    DfmModels,
    Cameras,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ControlKind {
    Toggle {
        default: bool,
    },
    Slider {
        min: f64,
        max: f64,
        default: f64,
        step: f64,
    },
    Choice {
        options: Vec<String>,
        default: String,
        #[serde(default)]
        source: Option<ChoiceSource>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
pub enum ControlValue {
    Toggle(bool),
    Slider(f64),
    Choice(String),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ControlState {
    values: BTreeMap<String, ControlValue>,
}

impl ControlState {
    pub fn from_catalog(catalog: &[ControlSpec]) -> Result<Self, ControlStateError> {
        let mut state = Self::default();
        for control in catalog {
            let value = match &control.kind {
                ControlKind::Toggle { default } => ControlValue::Toggle(*default),
                ControlKind::Slider { default, .. } => ControlValue::Slider(*default),
                ControlKind::Choice { default, .. } => ControlValue::Choice(default.clone()),
            };
            state.set(&control.id, value, catalog)?;
        }
        Ok(state)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&ControlValue> {
        self.values.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ControlValue)> {
        self.values.iter().map(|(id, value)| (id.as_str(), value))
    }

    pub fn validate_against(&self, catalog: &[ControlSpec]) -> Result<(), ControlStateError> {
        if self.values.len() != catalog.len() {
            return Err(ControlStateError::Incomplete {
                expected: catalog.len(),
                actual: self.values.len(),
            });
        }
        for (id, value) in &self.values {
            let mut probe = Self::default();
            probe.set(id, value.clone(), catalog)?;
        }
        Ok(())
    }

    pub fn apply_plain_json(
        &mut self,
        values: &serde_json::Map<String, serde_json::Value>,
        catalog: &[ControlSpec],
    ) -> Result<(), ControlStateError> {
        for (id, json) in values {
            let Some(control) = catalog.iter().find(|control| control.id == *id) else {
                continue;
            };
            let value = match &control.kind {
                ControlKind::Toggle { .. } => json.as_bool().map(ControlValue::Toggle),
                ControlKind::Slider { .. } => json.as_f64().map(ControlValue::Slider),
                ControlKind::Choice { .. } => json
                    .as_str()
                    .map(|value| ControlValue::Choice(value.to_owned())),
            }
            .ok_or_else(|| ControlStateError::InvalidPlainJson(id.clone()))?;
            self.set(id, value, catalog)?;
        }
        Ok(())
    }

    pub fn plain_json_for_scope(
        &self,
        scope: ControlScope,
        catalog: &[ControlSpec],
    ) -> serde_json::Map<String, serde_json::Value> {
        catalog
            .iter()
            .filter(|control| control.scope == scope)
            .filter_map(|control| {
                self.values
                    .get(&control.id)
                    .map(|value| (control.id.clone(), value.plain_json()))
            })
            .collect()
    }

    pub fn set(
        &mut self,
        id: &str,
        value: ControlValue,
        catalog: &[ControlSpec],
    ) -> Result<(), ControlStateError> {
        let control = catalog
            .iter()
            .find(|control| control.id == id)
            .ok_or_else(|| ControlStateError::UnknownControl(id.to_owned()))?;
        let valid = match (&control.kind, &value) {
            (ControlKind::Toggle { .. }, ControlValue::Toggle(_)) => true,
            (ControlKind::Slider { min, max, .. }, ControlValue::Slider(value)) => {
                value.is_finite() && min <= value && value <= max
            }
            (
                ControlKind::Choice {
                    options, source, ..
                },
                ControlValue::Choice(value),
            ) => options.contains(value) || (options.is_empty() && source.is_some()),
            _ => false,
        };
        if !valid {
            return Err(ControlStateError::InvalidValue {
                control: id.to_owned(),
                value,
            });
        }
        self.values.insert(id.to_owned(), value);
        Ok(())
    }

    pub fn is_visible(&self, id: &str, frontend: FrontendMode, catalog: &[ControlSpec]) -> bool {
        let Some(control) = catalog.iter().find(|control| control.id == id) else {
            return false;
        };
        if control.visibility.hidden_modes.contains(&frontend) {
            return false;
        }
        if control.visibility.dependencies.is_empty() {
            return true;
        }
        let matches = |dependency: &ControlDependency| match (
            self.values.get(&dependency.control),
            &dependency.value,
        ) {
            (Some(ControlValue::Toggle(actual)), DependencyValue::Toggle(required)) => {
                actual == required
            }
            (Some(ControlValue::Choice(actual)), DependencyValue::Choice(required)) => {
                actual == required
            }
            _ => false,
        };
        match control.visibility.mode {
            DependencyMode::Any => control.visibility.dependencies.iter().any(matches),
            DependencyMode::All => control.visibility.dependencies.iter().all(matches),
        }
    }
}

#[derive(Debug, Error)]
pub enum ControlStateError {
    #[error("unknown control {0}")]
    UnknownControl(String),
    #[error("invalid value for control {control}: {value:?}")]
    InvalidValue {
        control: String,
        value: ControlValue,
    },
    #[error("control state is incomplete: expected {expected}, got {actual}")]
    Incomplete { expected: usize, actual: usize },
    #[error("control {0} has an incompatible plain JSON value")]
    InvalidPlainJson(String),
}

impl ControlValue {
    fn plain_json(&self) -> serde_json::Value {
        match self {
            Self::Toggle(value) => serde_json::Value::Bool(*value),
            Self::Slider(value) => serde_json::Number::from_f64(*value)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            Self::Choice(value) => serde_json::Value::String(value.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum DependencyValue {
    Toggle(bool),
    Choice(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ControlDependency {
    pub control: String,
    pub value: DependencyValue,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DependencyMode {
    Any,
    #[default]
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrontendMode {
    Realtime,
    Editor,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Visibility {
    #[serde(default)]
    pub mode: DependencyMode,
    #[serde(default)]
    pub dependencies: Vec<ControlDependency>,
    #[serde(default)]
    pub hidden_modes: Vec<FrontendMode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ControlSpec {
    pub id: String,
    pub scope: ControlScope,
    pub section: String,
    pub label: String,
    pub level: u8,
    #[serde(default)]
    pub help: String,
    pub kind: ControlKind,
    #[serde(default)]
    pub visibility: Visibility,
}

impl ControlSpec {
    pub fn choice_options(&self) -> Option<Vec<&str>> {
        match &self.kind {
            ControlKind::Choice { options, .. } => {
                Some(options.iter().map(String::as_str).collect())
            }
            ControlKind::Toggle { .. } | ControlKind::Slider { .. } => None,
        }
    }
}

#[derive(Debug, Error)]
pub enum ControlCatalogError {
    #[error("invalid embedded control catalog: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("duplicate control id {0}")]
    DuplicateId(String),
    #[error("control {control} depends on missing control {dependency}")]
    MissingDependency { control: String, dependency: String },
    #[error("slider {0} has an invalid range/default/step")]
    InvalidSlider(String),
    #[error("choice {0} has an invalid default")]
    InvalidChoice(String),
}

pub fn control_catalog() -> Result<Vec<ControlSpec>, ControlCatalogError> {
    let controls: Vec<ControlSpec> = serde_json::from_str(include_str!("control_catalog.json"))?;
    validate(&controls)?;
    Ok(controls)
}

fn validate(controls: &[ControlSpec]) -> Result<(), ControlCatalogError> {
    let mut ids = HashSet::with_capacity(controls.len());
    for control in controls {
        if !ids.insert(control.id.as_str()) {
            return Err(ControlCatalogError::DuplicateId(control.id.clone()));
        }
        match &control.kind {
            ControlKind::Slider {
                min,
                max,
                default,
                step,
            } if !min.is_finite()
                || !max.is_finite()
                || !default.is_finite()
                || !step.is_finite()
                || min > default
                || default > max
                || *step <= 0.0 =>
            {
                return Err(ControlCatalogError::InvalidSlider(control.id.clone()));
            }
            ControlKind::Choice {
                options,
                default,
                source,
            } if (!options.is_empty() && !options.contains(default))
                || (options.is_empty() && source.is_none()) =>
            {
                return Err(ControlCatalogError::InvalidChoice(control.id.clone()));
            }
            ControlKind::Toggle { .. }
            | ControlKind::Slider { .. }
            | ControlKind::Choice { .. } => {}
        }
    }
    for control in controls {
        for dependency in &control.visibility.dependencies {
            if !ids.contains(dependency.control.as_str()) {
                return Err(ControlCatalogError::MissingDependency {
                    control: control.id.clone(),
                    dependency: dependency.control.clone(),
                });
            }
        }
    }
    Ok(())
}
