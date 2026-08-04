# Agent Phase Protocol

This directory defines the phase-gated execution protocol every coding agent must follow in this repository. It is binding for Phases 01–18.

## Every phase must have

- A single scoped objective.
- Explicit allowed files and explicit forbidden files.
- Required implementation artifacts.
- Unit tests, and integration tests where applicable.
- One smoke command and one acceptance command.
- A phase report (`reports/phase_<NN>/`).
- A machine-readable completion manifest (`acceptance.json`).
- No unresolved test failures, no silently skipped tests, no untracked architectural decisions.

## Workflow per phase

1. Read the phase specification in `implementation_plan/`, the handoff from the previous phase (`reports/phase_<N-1>/next_phase_handoff.md`), and the architecture docs (`docs/architecture/`).
2. Implement using `prompts/implement_phase.md`.
3. Test using `prompts/test_phase.md`.
4. Review using `prompts/review_phase.md`.
5. Repair using `prompts/repair_phase.md` until the gate passes.
6. Write the phase report and acceptance manifest; run the completion gate.

## Completion gate

Every phase must run something structurally equivalent to:

```bash
python -m pytest tests/phase_<NN> -q
python -m medfm.tools.validate_phase --phase <NN>
```

`validate_phase` verifies: required files exist; required tests ran; the phase report is populated; no acceptance criterion is `unknown`; no model license is missing; no dataset lacks provenance; no checkpoint lacks a base-model reference and configuration hash.

## Acceptance statuses

`acceptance.json` criteria statuses (schema: `acceptance_schema.json`):

- `passed` — criterion verified with recorded evidence.
- `failed` — criterion checked and not met; blocks phase completion.
- `blocked` — criterion cannot be checked due to an external dependency; blocks completion unless explicitly waived by the phase owner with a recorded reason.
- `not_applicable` — criterion does not apply to this phase; requires a justification string.

`unknown` is **not** a legal status. A phase is complete only when every criterion is `passed` or justifiably `not_applicable`.

## Required phase report files

```text
agent/reports/phase_<NN>/
├── summary.md               # what was done, key decisions
├── files_changed.txt        # one path per line
├── commands_executed.txt    # exact commands, one per line
├── test_results.json        # machine-readable test outcomes
├── acceptance.json          # per acceptance_schema.json
├── unresolved_issues.md     # known gaps, or explicit "none"
└── next_phase_handoff.md    # what the next phase needs
```

## Hard rules

- Modify only files the phase specification allows.
- Preserve public interfaces unless the phase explicitly changes them.
- Do not introduce an additional framework when existing dependencies suffice.
- Do not download patient data. Do not place model weights in Git.
- Do not fabricate test results. Do not proceed past a failing acceptance condition.
- **Any architectural decision discovered during implementation must be recorded as a tracked ADR** in `docs/architecture/` before the phase can close.
