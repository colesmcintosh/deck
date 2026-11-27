use std::{
    fs,
    path::{Path, PathBuf},
};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AdapterCandidate, AdapterKind, RetentionPolicy, TaskKind};

pub const DEFAULT_DECISIONS_ENDPOINT: &str = "https://openrouter.ai/api/alpha/decisions";
pub const DEFAULT_MODELS_CATALOG_URL: &str =
    "https://openrouter.ai/api/v1/models?output_modalities=decisions";
pub const DEFAULT_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
pub const DEFAULT_DECISION_TIMEOUT_MS: u64 = 8_000;
pub const REPO_CONFIG_FILENAME: &str = ".deck.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterTuning {
    pub adapter: AdapterKind,
    pub quality_tier: Option<u8>,
    pub speed_tier: Option<u8>,
    /// Higher values represent lower configured cost.
    pub cost_tier: Option<u8>,
}

/// Configuration for calling an OpenRouter Decisions (or System One-compatible) endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRouterConfig {
    #[serde(default = "default_decisions_endpoint")]
    pub endpoint: String,
    /// Pinned decision model ID (prefer a `canonical_slug` or versioned `id`, never an alias).
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_decision_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for DecisionRouterConfig {
    fn default() -> Self {
        Self {
            endpoint: default_decisions_endpoint(),
            model: None,
            api_key_env: default_api_key_env(),
            timeout_ms: default_decision_timeout_ms(),
        }
    }
}

impl DecisionRouterConfig {
    /// Resolves the active decision model ID using precedence:
    /// CLI override -> `DECK_DECISION_MODEL` -> `DECISION_MODEL` -> config file.
    pub fn resolved_model(&self, cli_override: Option<&str>) -> Option<String> {
        if let Some(value) = cli_override
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(value.to_owned());
        }
        for env_name in ["DECK_DECISION_MODEL", "DECISION_MODEL"] {
            if let Ok(value) = std::env::var(env_name) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_owned());
                }
            }
        }
        self.model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    pub fn is_alias(model: &str) -> bool {
        model.trim().starts_with('~')
    }

    pub fn api_key(&self) -> Option<String> {
        std::env::var(&self.api_key_env)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    }

    pub fn has_api_key(&self) -> bool {
        self.api_key().is_some()
    }
}

/// One selectable `(harness + model)` configuration in the user's deck.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentCard {
    /// Descriptive identifier used as the option name in decision questions (never an index).
    pub id: String,
    pub adapter: AdapterKind,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub quality_tier: Option<u8>,
    #[serde(default)]
    pub speed_tier: Option<u8>,
    /// Higher values represent lower configured cost.
    #[serde(default)]
    pub cost_tier: Option<u8>,
    /// Empty means eligible for every task role.
    #[serde(default)]
    pub allowed_roles: Vec<TaskKind>,
}

impl AgentCard {
    pub fn allows_role(&self, role: TaskKind) -> bool {
        self.allowed_roles.is_empty() || self.allowed_roles.contains(&role)
    }

    pub fn effective_description(&self) -> String {
        let trimmed = self.description.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
        default_adapter_strengths(self.adapter).to_owned()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRouterOverride {
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoDeckOverride {
    #[serde(default)]
    pub provider_priority: Option<Vec<AdapterKind>>,
    #[serde(default)]
    pub adapter_tuning: Vec<AdapterTuning>,
    #[serde(default)]
    pub decision_router: Option<DecisionRouterOverride>,
    #[serde(default)]
    pub cards: Vec<AgentCard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigSources {
    pub global_path: PathBuf,
    pub global_loaded: bool,
    pub repo_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeckConfig {
    #[serde(default = "default_provider_priority")]
    pub provider_priority: Vec<AdapterKind>,
    #[serde(default = "default_adapter_tuning")]
    pub adapter_tuning: Vec<AdapterTuning>,
    #[serde(default = "default_retention")]
    pub retention: RetentionPolicy,
    #[serde(default)]
    pub decision_router: DecisionRouterConfig,
    #[serde(default)]
    pub cards: Vec<AgentCard>,
}

impl Default for DeckConfig {
    fn default() -> Self {
        Self {
            provider_priority: default_provider_priority(),
            adapter_tuning: default_adapter_tuning(),
            retention: default_retention(),
            decision_router: DecisionRouterConfig::default(),
            cards: Vec::new(),
        }
    }
}

impl DeckConfig {
    pub fn load_default() -> Result<(Self, PathBuf), ConfigError> {
        let directories =
            ProjectDirs::from("dev", "deck", "Deck").ok_or(ConfigError::NoApplicationDirectory)?;
        let path = directories.config_dir().join("config.json");
        if !path.exists() {
            return Ok((Self::default(), path));
        }
        let config = Self::load_from_path(&path)?;
        Ok((config, path))
    }

    pub fn load_for_repo(repository: &Path) -> Result<(Self, ConfigSources), ConfigError> {
        let directories =
            ProjectDirs::from("dev", "deck", "Deck").ok_or(ConfigError::NoApplicationDirectory)?;
        let global_path = directories.config_dir().join("config.json");
        let (mut config, global_loaded) = if global_path.exists() {
            (Self::load_from_path(&global_path)?, true)
        } else {
            (Self::default(), false)
        };
        let candidate_repo_path = repository.join(REPO_CONFIG_FILENAME);
        let repo_path = if candidate_repo_path.exists() {
            let repo_override = load_repo_override(&candidate_repo_path)?;
            config.merge_repo_override(repo_override);
            Some(candidate_repo_path)
        } else {
            None
        };
        Ok((
            config,
            ConfigSources {
                global_path,
                global_loaded,
                repo_path,
            },
        ))
    }

    pub fn load_from_path(path: &Path) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&contents).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn merge_repo_override(&mut self, repo_override: RepoDeckOverride) {
        if let Some(priority) = repo_override.provider_priority {
            self.provider_priority = priority;
        }
        for tuning in repo_override.adapter_tuning {
            if let Some(existing) = self
                .adapter_tuning
                .iter_mut()
                .find(|entry| entry.adapter == tuning.adapter)
            {
                *existing = tuning;
            } else {
                self.adapter_tuning.push(tuning);
            }
        }
        if let Some(router) = repo_override.decision_router {
            if let Some(endpoint) = router.endpoint {
                self.decision_router.endpoint = endpoint;
            }
            if let Some(model) = router.model {
                self.decision_router.model = Some(model);
            }
            if let Some(api_key_env) = router.api_key_env {
                self.decision_router.api_key_env = api_key_env;
            }
            if let Some(timeout_ms) = router.timeout_ms {
                self.decision_router.timeout_ms = timeout_ms;
            }
        }
        for card in repo_override.cards {
            if let Some(existing) = self.cards.iter_mut().find(|entry| entry.id == card.id) {
                *existing = card;
            } else {
                self.cards.push(card);
            }
        }
    }

    pub fn tuning(&self, adapter: AdapterKind) -> Option<&AdapterTuning> {
        self.adapter_tuning
            .iter()
            .find(|tuning| tuning.adapter == adapter)
    }

    /// Returns user-defined cards (with missing tier metadata inherited from adapter tuning),
    /// or synthesizes one default card per installed candidate when `cards` is empty.
    pub fn effective_cards(&self, candidates: &[AdapterCandidate]) -> Vec<AgentCard> {
        if !self.cards.is_empty() {
            return self
                .cards
                .iter()
                .map(|card| {
                    let tuning = self.tuning(card.adapter);
                    let mut resolved = card.clone();
                    if resolved.quality_tier.is_none() {
                        resolved.quality_tier = tuning.and_then(|entry| entry.quality_tier);
                    }
                    if resolved.speed_tier.is_none() {
                        resolved.speed_tier = tuning.and_then(|entry| entry.speed_tier);
                    }
                    if resolved.cost_tier.is_none() {
                        resolved.cost_tier = tuning.and_then(|entry| entry.cost_tier);
                    }
                    resolved
                })
                .collect();
        }
        candidates
            .iter()
            .filter(|candidate| candidate.installed && candidate.ready)
            .map(|candidate| {
                let tuning = self.tuning(candidate.kind);
                AgentCard {
                    id: default_card_id(candidate.kind).to_owned(),
                    adapter: candidate.kind,
                    model: None,
                    description: default_adapter_strengths(candidate.kind).to_owned(),
                    quality_tier: candidate
                        .quality_tier
                        .or_else(|| tuning.and_then(|entry| entry.quality_tier)),
                    speed_tier: candidate
                        .speed_tier
                        .or_else(|| tuning.and_then(|entry| entry.speed_tier)),
                    cost_tier: candidate
                        .cost_tier
                        .or_else(|| tuning.and_then(|entry| entry.cost_tier)),
                    allowed_roles: Vec::new(),
                }
            })
            .collect()
    }
}

fn load_repo_override(path: &Path) -> Result<RepoDeckOverride, ConfigError> {
    let contents = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&contents).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

fn default_provider_priority() -> Vec<AdapterKind> {
    AdapterKind::ALL.to_vec()
}

fn default_adapter_tuning() -> Vec<AdapterTuning> {
    AdapterKind::ALL
        .into_iter()
        .map(|adapter| AdapterTuning {
            adapter,
            quality_tier: None,
            speed_tier: None,
            cost_tier: None,
        })
        .collect()
}

const fn default_retention() -> RetentionPolicy {
    RetentionPolicy::KeepForever
}

fn default_decisions_endpoint() -> String {
    DEFAULT_DECISIONS_ENDPOINT.to_owned()
}

fn default_api_key_env() -> String {
    DEFAULT_API_KEY_ENV.to_owned()
}

const fn default_decision_timeout_ms() -> u64 {
    DEFAULT_DECISION_TIMEOUT_MS
}

const fn default_card_id(adapter: AdapterKind) -> &'static str {
    match adapter {
        AdapterKind::Claude => "claude",
        AdapterKind::Codex => "codex",
        AdapterKind::OpenCode => "open-code",
        AdapterKind::Cursor => "cursor",
        AdapterKind::Grok => "grok",
        AdapterKind::Droid => "droid",
    }
}

const fn default_adapter_strengths(adapter: AdapterKind) -> &'static str {
    match adapter {
        AdapterKind::Claude => {
            "Claude Code CLI harness for architectural reasoning, refactoring, code review, and multi-file edits."
        }
        AdapterKind::Codex => {
            "Codex CLI harness for autonomous implementation, test-driven coding, and backend or CLI workflows."
        }
        AdapterKind::OpenCode => {
            "OpenCode CLI harness for general multi-provider repository analysis and scoped workspace edits."
        }
        AdapterKind::Cursor => {
            "Cursor agent CLI harness for interactive workspace edits and frontend or full-stack implementation."
        }
        AdapterKind::Grok => {
            "Grok Build CLI harness for fast repository inspection, troubleshooting, and implementation tasks."
        }
        AdapterKind::Droid => {
            "Factory Droid CLI harness for structured execution, automation, and repository engineering tasks."
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot determine the local application configuration directory")]
    NoApplicationDirectory,
    #[error("cannot read configuration {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot parse configuration {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn defaults_do_not_invent_provider_rankings() {
        let config = DeckConfig::default();
        assert!(config.adapter_tuning.iter().all(|tuning| {
            tuning.quality_tier.is_none()
                && tuning.speed_tier.is_none()
                && tuning.cost_tier.is_none()
        }));
        assert_eq!(config.retention, RetentionPolicy::KeepForever);
        assert_eq!(config.decision_router.endpoint, DEFAULT_DECISIONS_ENDPOINT);
        assert!(config.decision_router.model.is_none());
        assert_eq!(config.cards.len(), 0);
    }

    #[test]
    fn partial_config_and_repo_override_merge_cards_by_id() {
        let directory = tempdir().unwrap();
        let global_path = directory.path().join("config.json");
        fs::write(
            &global_path,
            r#"{
              "decision_router": {
                "model": "typesafe/jev-1.13-20260917"
              },
              "cards": [
                {
                  "id": "claude-opus",
                  "adapter": "claude",
                  "model": "claude-opus-4-6",
                  "description": "Architecture and review"
                },
                {
                  "id": "codex-main",
                  "adapter": "codex",
                  "model": "gpt-5-codex",
                  "description": "Implementation"
                }
              ]
            }"#,
        )
        .unwrap();

        let mut config = DeckConfig::load_from_path(&global_path).unwrap();
        assert_eq!(
            config.decision_router.model.as_deref(),
            Some("typesafe/jev-1.13-20260917")
        );
        assert_eq!(config.cards.len(), 2);

        let repo_override: RepoDeckOverride = serde_json::from_str(
            r#"{
              "decision_router": {
                "timeout_ms": 5000
              },
              "cards": [
                {
                  "id": "codex-main",
                  "adapter": "codex",
                  "model": "gpt-5-codex-mini",
                  "description": "Fast repo tests"
                },
                {
                  "id": "droid-ci",
                  "adapter": "droid",
                  "description": "CI scripting"
                }
              ]
            }"#,
        )
        .unwrap();

        config.merge_repo_override(repo_override);
        assert_eq!(config.decision_router.timeout_ms, 5000);
        assert_eq!(
            config.decision_router.model.as_deref(),
            Some("typesafe/jev-1.13-20260917")
        );
        assert_eq!(config.cards.len(), 3);
        let updated = config
            .cards
            .iter()
            .find(|card| card.id == "codex-main")
            .unwrap();
        assert_eq!(updated.model.as_deref(), Some("gpt-5-codex-mini"));
        assert_eq!(updated.description, "Fast repo tests");
    }

    #[test]
    fn detects_model_aliases() {
        assert!(DecisionRouterConfig::is_alias("~typesafe/jev-latest"));
        assert!(!DecisionRouterConfig::is_alias("typesafe/jev-1.14"));
    }
}
