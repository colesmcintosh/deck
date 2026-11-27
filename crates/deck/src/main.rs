mod runner;
mod tui;

use std::{path::PathBuf, time::Duration};

use clap::{Parser, Subcommand, ValueEnum};
use deck_adapters::{ProbeReadiness, ProbeResult, adapters};
use deck_core::{
    AdapterCandidate, AutonomyPolicy, CardDealInput, CardDealResult, ConfigSources,
    DecisionModelInfo, DecisionRouterStatus, DeckConfig, HistoryStore, RepositoryStatus,
    RoutingObjective, SkillIndex, TaskKind, WorkspaceManager, deal_card, list_decision_models,
};
use serde::Serialize;
use tokio::task::JoinSet;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "deck",
    version,
    about = "One local CLI for your deck of coding agents — deal the best hand for every job"
)]
struct Cli {
    /// Repository or workspace to orchestrate.
    #[arg(long, global = true, value_name = "PATH")]
    repo: Option<PathBuf>,

    /// Force ASCII rail characters for limited terminals.
    #[arg(long, global = true)]
    ascii: bool,

    /// Override the pinned decision model ID (e.g. a canonical_slug from `deck models`).
    #[arg(long, global = true, value_name = "MODEL")]
    decision_model: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliObjective {
    Balanced,
    Quality,
    Speed,
    Cost,
}

impl From<CliObjective> for RoutingObjective {
    fn from(value: CliObjective) -> Self {
        match value {
            CliObjective::Balanced => Self::Balanced,
            CliObjective::Quality => Self::Quality,
            CliObjective::Speed => Self::Speed,
            CliObjective::Cost => Self::Cost,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliRole {
    Analysis,
    Implementation,
    Review,
    Integration,
}

impl From<CliRole> for TaskKind {
    fn from(value: CliRole) -> Self {
        match value {
            CliRole::Analysis => Self::Analysis,
            CliRole::Implementation => Self::Implementation,
            CliRole::Review => Self::Review,
            CliRole::Integration => Self::Integration,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliAutonomy {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

impl From<CliAutonomy> for AutonomyPolicy {
    fn from(value: CliAutonomy) -> Self {
        match value {
            CliAutonomy::ReadOnly => Self::ReadOnly,
            CliAutonomy::WorkspaceWrite => Self::WorkspaceWrite,
            CliAutonomy::FullAccess => Self::FullAccess,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect Git, coding-agent CLIs, and decision-router configuration without invoking a model.
    Doctor {
        /// Emit one machine-readable JSON document.
        #[arg(long)]
        json: bool,
    },
    /// Pick the best (harness + model) card for a task using your configured decision model.
    Deal {
        /// Task or goal description to route.
        task: String,

        /// Optional operating constraints attached to the task.
        #[arg(long = "constraint", value_name = "TEXT")]
        constraints: Vec<String>,

        /// Routing priority used to break close ties or balance cost/speed on routine tasks.
        #[arg(long, value_enum, default_value_t = CliObjective::Balanced)]
        objective: CliObjective,

        /// Optional task role filter (`analysis`, `implementation`, `review`, `integration`).
        #[arg(long, value_enum)]
        role: Option<CliRole>,

        /// Optional autonomy policy filter (`read-only`, `workspace-write`, `full-access`).
        #[arg(long, value_enum)]
        autonomy: Option<CliAutonomy>,

        /// Skip calling the decision model and use deterministic ranking.
        #[arg(long)]
        no_decision_model: bool,

        /// Emit one machine-readable JSON document.
        #[arg(long)]
        json: bool,
    },
    /// List live decision models currently served by OpenRouter (`output_modalities=decisions`).
    Models {
        /// Emit one machine-readable JSON document.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    repository: RepositoryStatus,
    adapters: Vec<ProbeResult>,
    decision_router: DecisionRouterStatus,
    config_sources: ConfigSources,
    notes: Vec<&'static str>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init()
        .ok();
    let cli = Cli::parse();

    if let Some(Command::Models { json }) = cli.command {
        let models = list_decision_models(Duration::from_secs(15)).await?;
        if json {
            println!("{}", serde_json::to_string_pretty(&models)?);
        } else {
            print_models(&models);
        }
        return Ok(());
    }

    let repository = cli.repo.unwrap_or(std::env::current_dir()?);
    let repository = repository.canonicalize().unwrap_or(repository);
    let repository_status = WorkspaceManager::inspect(&repository)?;
    let (config, config_sources) = DeckConfig::load_for_repo(&repository)?;
    let probes = probe_all().await;

    if let Some(Command::Doctor { json }) = cli.command {
        let report = DoctorReport {
            repository: repository_status,
            adapters: probes,
            decision_router: DecisionRouterStatus::inspect(&config, cli.decision_model.as_deref()),
            config_sources,
            notes: vec![
                "Version probes never invoke a paid model.",
                "Authentication remains provider-managed and is verified when a run starts.",
                "Decision routing is opt-in and only runs when a decision model and API key are configured.",
                "Deck has no telemetry or remote history synchronization.",
            ],
        };
        if json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_doctor(&report);
        }
        return Ok(());
    }

    if let Some(Command::Deal {
        task,
        constraints,
        objective,
        role,
        autonomy,
        no_decision_model,
        json,
    }) = cli.command
    {
        let candidates = build_candidates(&probes, &config);
        let result = deal_card(
            &config,
            &candidates,
            &CardDealInput {
                task: &task,
                constraints: &constraints,
                repository: &repository,
                objective: objective.into(),
                role: role.map(Into::into),
                autonomy: autonomy.map(Into::into),
                preferred_adapters: &[],
                decision_model_override: cli.decision_model.as_deref(),
                use_decision_model: !no_decision_model,
            },
        )
        .await?;
        if json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            print_deal_result(&result);
        }
        return Ok(());
    }

    let history = HistoryStore::open_default()?;
    history.mark_unfinished_interrupted()?;
    history.apply_retention(config.retention)?;
    let user_skill_locations = std::env::var_os("HOME").map_or_else(Vec::new, |home| {
        let home = PathBuf::from(home);
        vec![
            home.join(".agents/skills"),
            home.join(".codex/skills"),
            home.join(".claude/skills"),
            home.join(".grok/skills"),
            home.join(".factory/skills"),
            home.join(".config/opencode/skills"),
        ]
    });
    let skills = SkillIndex::discover(&repository, &user_skill_locations);
    let mut app = tui::App::new(
        repository_status,
        probes,
        skills,
        history,
        config,
        cli.decision_model,
        cli.ascii,
    );
    tui::run(&mut app)?;
    Ok(())
}

fn build_candidates(probes: &[ProbeResult], config: &DeckConfig) -> Vec<AdapterCandidate> {
    let mut candidates = probes
        .iter()
        .map(|probe| {
            let tuning = config.tuning(probe.adapter);
            AdapterCandidate {
                kind: probe.adapter,
                installed: probe.installed,
                ready: probe.readiness != ProbeReadiness::Unavailable,
                capabilities: probe.capabilities.clone(),
                quality_tier: tuning.and_then(|entry| entry.quality_tier),
                speed_tier: tuning.and_then(|entry| entry.speed_tier),
                cost_tier: tuning.and_then(|entry| entry.cost_tier),
                historical_duration_ms: None,
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| {
        config
            .provider_priority
            .iter()
            .position(|adapter| *adapter == candidate.kind)
            .unwrap_or(usize::MAX)
    });
    candidates
}

async fn probe_all() -> Vec<ProbeResult> {
    let mut joins = JoinSet::new();
    for adapter in adapters() {
        joins.spawn(async move { adapter.probe().await });
    }
    let mut probes = Vec::new();
    while let Some(result) = joins.join_next().await {
        if let Ok(probe) = result {
            probes.push(probe);
        }
    }
    probes.sort_by_key(|probe| {
        deck_core::AdapterKind::ALL
            .iter()
            .position(|kind| *kind == probe.adapter)
            .unwrap_or(usize::MAX)
    });
    probes
}

fn print_doctor(report: &DoctorReport) {
    println!(
        "Repository: {} [{}]",
        report.repository.path.display(),
        if report.repository.is_git {
            if report.repository.dirty {
                "Git, dirty"
            } else {
                "Git, clean"
            }
        } else {
            "non-Git"
        }
    );
    for probe in &report.adapters {
        println!(
            "[{:<4}] {:<14} {:<24} structured={} resume={} model={} readiness={:?}",
            if probe.installed { "OK" } else { "MISS" },
            probe.adapter,
            probe.version.as_deref().unwrap_or("not installed"),
            probe.capabilities.structured_output,
            probe.capabilities.resume,
            probe.capabilities.model_selection,
            probe.readiness,
        );
    }
    let router = &report.decision_router;
    println!(
        "[{:<4}] Decision Router  model={}  key({}){}  cards={}  endpoint={}",
        if router.active { "OK" } else { "OFF" },
        router.configured_model.as_deref().unwrap_or("unconfigured"),
        router.api_key_env,
        if router.api_key_present {
            "=set"
        } else {
            "=unset"
        },
        if router.configured_cards == 0 {
            "auto".to_owned()
        } else {
            router.configured_cards.to_string()
        },
        router.endpoint,
    );
    if router.is_alias {
        println!(
            "WARN: Configured decision model is an alias; pin a canonical_slug from `deck models`."
        );
    }
    if let Some(repo_cfg) = &report.config_sources.repo_path {
        println!("Repo config: {}", repo_cfg.display());
    }
    for note in &report.notes {
        println!("NOTE: {note}");
    }
}

fn print_deal_result(result: &CardDealResult) {
    println!(
        "Selected card: {}  (harness={}, model={})",
        result.selected.id,
        result.selected.adapter,
        result
            .selected
            .model
            .as_deref()
            .unwrap_or("harness default"),
    );
    println!("Source:        {:?}", result.source);
    println!("Reason:        {}", result.reason);
    if let Some(model) = &result.resolved_model {
        println!(
            "Decision build: {model}{}{}",
            result
                .latency_ms
                .map_or_else(String::new, |ms| format!("  ({ms}ms)")),
            result
                .cost_usd
                .map_or_else(String::new, |cost| format!("  (${cost:.6})")),
        );
    }
    if let Some(fit) = result.has_clear_fit {
        println!("Clear fit:     {fit:.2}");
    }
    if let Some(score) = result.complexity_score {
        let dist = result
            .complexity_probabilities
            .as_ref()
            .map_or_else(String::new, |probs| {
                probs
                    .iter()
                    .map(|(level, prob)| format!("L{level}={prob:.2}"))
                    .collect::<Vec<_>>()
                    .join("  ")
            });
        println!("Complexity:    {score:.2}  {dist}");
    }
    if !result.card_probabilities.is_empty() {
        let mut pairs: Vec<(&String, f64)> = result
            .card_probabilities
            .iter()
            .map(|(id, prob)| (id, *prob))
            .collect();
        pairs.sort_by(|left, right| right.1.total_cmp(&left.1));
        println!("Probabilities:");
        for (id, prob) in pairs {
            let marker = if *id == result.selected.id { "*" } else { " " };
            println!("  {marker} {id:<20} {prob:.3}");
        }
    }
    for warning in &result.warnings {
        println!("WARN: {warning}");
    }
}

fn print_models(models: &[DecisionModelInfo]) {
    if models.is_empty() {
        println!("No decision models were returned by OpenRouter.");
        return;
    }
    println!(
        "{:<34} {:<36} {:>9} {:>10}  KIND",
        "ID", "PINNED CANONICAL_SLUG", "CONTEXT", "$/1M IN"
    );
    for model in models {
        println!(
            "{:<34} {:<36} {:>9} {:>10.4}  {}",
            model.id,
            model.canonical_slug,
            model.context_length,
            model.prompt_price_per_million,
            if model.is_alias() { "alias" } else { "pinned" }
        );
    }
    println!("NOTE: Pin a `canonical_slug` (not an alias) in config.json or .deck.json.");
}
