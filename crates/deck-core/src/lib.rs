//! Provider-neutral contracts and routing primitives for Deck.

pub mod config;
pub mod decision;
pub mod model;
pub mod persistence;
pub mod planner;
pub mod scheduler;
pub mod skills;
pub mod state;
pub mod validation;
pub mod workspace;

pub use config::{
    AdapterTuning, AgentCard, ConfigError, ConfigSources, DEFAULT_API_KEY_ENV,
    DEFAULT_DECISION_TIMEOUT_MS, DEFAULT_DECISIONS_ENDPOINT, DEFAULT_MODELS_CATALOG_URL,
    DecisionRouterConfig, DecisionRouterOverride, DeckConfig, REPO_CONFIG_FILENAME,
    RepoDeckOverride,
};
pub use decision::{
    CHOICE_CLEAR_MARGIN, CardDealInput, CardDealResult, DealtHand, DecisionAnswer, DecisionClient,
    DecisionError, DecisionModelInfo, DecisionQuestion, DecisionRouterStatus, DecisionsRequest,
    DecisionsResponse, DecisionsUsage, HAS_FIT_NO_THRESHOLD, HAS_FIT_YES_THRESHOLD,
    HIGH_COMPLEXITY_PROB_THRESHOLD, HandDecisionSummary, NoulCriteria,
    ROUTINE_COMPLEXITY_PROB_THRESHOLD, RoleDealSummary, SelectionSource,
    VIABLE_CARD_PROB_THRESHOLD, build_role_deal_request, deal_card, deal_hand_with_decisions,
    detect_repository_stack, list_decision_models, list_decision_models_from, redact_decision_text,
    validate_decisions_response,
};
pub use model::*;
pub use persistence::{HistoryStore, RetentionPolicy};
pub use planner::{AdapterCandidate, RecommendationEngine, TeamPlanValidationError, validate_plan};
pub use scheduler::{ScheduleError, TaskExecution, TaskRunner, execute_plan};
pub use skills::{SkillIndex, SkillMetadata};
pub use state::{StateTransitionError, transition_agent, transition_run};
pub use validation::{ValidationError, run_acceptance_checks};
pub use workspace::{RepositoryStatus, WorkspaceError, WorkspaceManager, WorkspaceSet};
