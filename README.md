# Deck

[![CI](https://github.com/colesmcintosh/deck/actions/workflows/ci.yml/badge.svg)](https://github.com/colesmcintosh/deck/actions/workflows/ci.yml)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-DEA584?logo=rust)](https://www.rust-lang.org/)
[![No telemetry](https://img.shields.io/badge/telemetry-none-6FB98F)](#configuration-and-local-data)

Deck is one local CLI for your deck of coding agents. Give it a goal and it helps choose and deal the best hand—assigning agents, models, a task DAG, Agent Skills, workspace controls, and executable quality gates—then routes the run through planning, parallel work, verification, bounded rework, and staged integration. Every hand can be edited before launch and saved as a reusable blueprint.

It is designed for developers who already install and authenticate their coding-agent CLIs. Deck does not embed provider SDKs, collect API keys, modify provider-global configuration, merge into the starting branch, send telemetry, or synchronize run history.

## Status

This repository contains a runnable first release for macOS and Linux. It includes the Ratatui switchboard TUI, six built-in CLI adapters, deterministic fallback planning, provider-neutral contracts, bounded DAG scheduling, worktree isolation and binary patch assembly, direct acceptance checks, process-group cancellation, local SQLite history, Agent Skills discovery, and reusable blueprints.

Planner output validation and repair are exposed by `RecommendationEngine`; the interactive path currently uses its deterministic fallback planner so dealing a hand never spends provider quota. The adapter boundary supports plugging in a structured provider planner without changing the scheduler or TUI.

## Install and run

Rust 1.85 or newer is required.

```sh
cargo install --path crates/deck --locked
deck doctor
deck doctor --json
deck models
deck deal "Refactor the scheduler to avoid deadlocks"
deck --repo /path/to/repository
```

Running `deck` opens the TUI. Its normal flow is:

1. Inspect Git status and the coding-agent CLIs in your deck.
2. Enter the goal, hand size (worker cells), parallel-lane count, routing priority, provider preferences, and operating constraints.
3. Edit the dealt hand: agents, provider/model assignments, objectives, dependencies, skills, and quality commands (including argument arrays, timeouts, and required/optional status).
4. Explicitly select workspace and autonomy controls.
5. Play the hand and watch the live routing rail.
6. Inspect the result or save the hand as a reusable blueprint.

The interface works at 80×24, uses color-independent status labels, supports `--ascii`, and exposes the launch control to both Enter and mouse click. `Ctrl-C` during a run cancels process groups while preserving logs and workspaces; outside a run it exits.

## Safety model

Every run requires two independent choices. There are no executable defaults.

Workspace policy:

- **Worktree per agent** starts each worker from the selected committed `HEAD`, detects overlapping paths, applies binary-safe patches into a dedicated integration worktree, and stages successful output there.
- **Shared workspace** operates in the selected directory. Existing changes are preserved, but attribution, collision handling, and rollback are limited.
- **Read-only then executor** keeps parallel cells read-only and gives only the final executor a dedicated integration worktree.

Autonomy policy:

- **Read-only** maps to the provider's supported read/plan controls.
- **Workspace write** maps only to documented workspace-scoped or edit-accepting controls.
- **Full access** can expose dangerous native bypass flags and requires a second `!` confirmation for each run.

Unsupported provider/policy combinations are blocked before launch. Prompts are passed as process arguments, never interpolated into shell strings. Standard output and error remain separate, lines are capped at 1 MiB, unknown JSON becomes a redacted diagnostic event, and credential-shaped values are removed from diagnostics. Deck inherits the environment needed by an already-authenticated CLI but never reads provider credential files.

Dirty repositories are explicit: isolated modes use committed `HEAD` and exclude uncommitted changes; shared mode retains them and warns about limited attribution. Non-Git directories can use only shared mode. Successful isolated runs remain on a `deck/<run>/integration` branch and worktree; Deck never merges it into the starting branch.

## Provider contracts

The built-ins probe the executable and version without invoking a paid model. Authentication is provider-owned and is first exercised when a task starts.

| Adapter | Executable | Structured command | Resume | Policy notes |
|---|---|---|---|---|
| Claude Code | `claude` | `--print --verbose --output-format stream-json` | `--resume` | plan, accept-edits, or separately confirmed bypass |
| Codex | `codex` | `exec --json` | `exec … resume` | explicit sandbox and approval flags; no app server |
| OpenCode | `opencode` | `run --format json` | `--session` | plan agent for read-only; `--auto` for writes |
| Cursor | `agent` | `--print --output-format stream-json` | `--resume` | no enforceable read-only mapping; force requires confirmation |
| Grok Build | `grok` | `--single … --output-format streaming-json` | `--resume` | documented permission modes and sandbox profiles |
| Factory Droid | `droid` | `exec --output-format stream-json` | `--session-id` | read-only default, tiered auto, or confirmed unsafe bypass |

The command shapes follow the providers' published headless references: [Claude Code](https://docs.anthropic.com/en/docs/claude-code/cli-usage), [Codex](https://developers.openai.com/codex/cli/reference), [OpenCode](https://dev.opencode.ai/docs/cli/), [Cursor](https://docs.cursor.com/en/cli/reference/output-format), [Grok Build](https://docs.x.ai/build/cli/headless-scripting), and [Factory Droid](https://docs.factory.ai/droid-exec/overview). Provider releases can change; use `deck doctor --json` to capture the installed version and advertised Deck capabilities when reporting an incompatibility.

Pi and Amp are intentionally not built in yet.

## Configuration, decision routing, and local data

Deck resolves platform paths with `directories::ProjectDirs("dev", "deck", "Deck")`. `config.json` lives in the resulting OS configuration directory and `history.sqlite3` in the OS data directory (XDG locations on Linux and Application Support on macOS). SQLite uses WAL mode. An optional repo-local `.deck.json` overlays global `config.json` (merging `cards` by `id` and overriding `decision_router` fields).

Configuration is optional. When no decision model or API key is configured, Deck deals hands deterministically and never makes network calls. To route tasks with a user-specified decision model via OpenRouter's Decisions API (or any System One-compatible endpoint):

1. Inspect live decision models and pick a pinned `canonical_slug` (never an alias):
   ```sh
   deck models
   ```
2. Export `OPENROUTER_API_KEY` and configure `decision_router` plus your `(harness + model)` `cards` in `config.json` or repo `.deck.json`:

```json
{
  "provider_priority": ["codex", "claude", "open-code", "cursor", "grok", "droid"],
  "decision_router": {
    "endpoint": "https://openrouter.ai/api/alpha/decisions",
    "model": "<pinned canonical_slug from `deck models`>",
    "api_key_env": "OPENROUTER_API_KEY",
    "timeout_ms": 8000
  },
  "cards": [
    {
      "id": "claude-opus",
      "adapter": "claude",
      "model": "claude-opus-4-6",
      "description": "Deep architectural reasoning, concurrency debugging, security review, and complex refactors.",
      "quality_tier": 5,
      "speed_tier": 2,
      "cost_tier": 1
    },
    {
      "id": "codex-main",
      "adapter": "codex",
      "model": "gpt-5-codex",
      "description": "Autonomous implementation, backend/CLI engineering, and test-driven coding.",
      "quality_tier": 4,
      "speed_tier": 4,
      "cost_tier": 3
    },
    {
      "id": "droid-fast",
      "adapter": "droid",
      "model": "claude-haiku-4-5",
      "description": "Fast, low-cost docs tweaks, copy edits, and localized single-file fixes.",
      "quality_tier": 2,
      "speed_tier": 5,
      "cost_tier": 5
    }
  ],
  "retention": { "keep-latest": 250 }
}
```

Override the pinned decision model per invocation with `--decision-model <id>` or `DECK_DECISION_MODEL` / `DECISION_MODEL`. When a task is uncertain, off-topic, offline, or hits a transport/validation error, Deck falls back to its deterministic planner and displays the raw probabilities or fallback reason. Representative probe payloads live in `probes/` so you can probe and recalibrate thresholds whenever you switch decision models.

History is kept forever by default. A run records its goal, plan, assignments, state transitions, normalized events, transcript content, checks, artifacts, probes, and worktree metadata. The history screen can select, export, delete, or load a prior blueprint for a clean retry. Deleting history does not delete review worktrees. On restart, unfinished runs become `Interrupted` and retain their artifacts. Adapter contracts preserve provider-specific resume invocation support for a future task-level resume flow; the current TUI deliberately offers a clean retry.

Agent Skills are indexed from repository `SKILL.md` files and the user's `.agents/skills` and `.codex/skills` directories. Selected instructions are copied only into the run-scoped prompt bundle. Native provider discovery continues unchanged; Deck does not install skills or rewrite global provider settings.

## Architecture and extension boundary

The Cargo workspace has three crates:

- `deck-core`: stable domain contracts, recommendation and validation, state machines, scheduler, workspaces, checks, configuration, skills, and SQLite history.
- `deck-adapters`: `AgentAdapter`, direct invocations, policy maps, stream normalization, redaction, probes, and process supervision.
- `deck`: command parsing, execution composition, and the switchboard TUI.

To add a Rust adapter:

1. Add its `AdapterKind`, executable, capability declaration, documented invocation, resume form, and policy mapping in `deck-adapters`.
2. Normalize only stable event fields; preserve unknown objects as diagnostics.
3. Add JSONL fixtures for init, assistant text, tools, usage, success, failure, malformed lines, stderr, unknown fields, and session IDs.
4. Add command-construction and policy tests. Never infer a bypass flag.
5. Update `deck doctor` output and this table.

`AgentAdapter` is deliberately the future generic-adapter boundary: the scheduler sees only `TaskNode`, `AgentEvent`, and `TaskExecution`. Adding Pi, Amp, ACP, or a declarative adapter does not require scheduler or TUI changes.

## Development and verification

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

Default tests use recorded streams, temporary Git repositories, and fake executables; they never invoke a paid model. A deliberately ignored smoke test is available only with both an explicit environment flag and `--ignored`:

```sh
DECK_LIVE_TESTS=1 cargo test -p deck-adapters --test live_providers -- --ignored
```

That command can consume provider quota. It runs installed adapters read-only in a temporary directory and depends on the user's existing authentication.

No license file is included; licensing remains an explicit project decision.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions, [SECURITY.md](SECURITY.md) for private vulnerability reporting, and [CHANGELOG.md](CHANGELOG.md) for release notes.
