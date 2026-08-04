# Implement Phase Prompt

You are implementing Phase <N> of the medical foundation-model framework.

Read:

1. `docs/architecture/*` (all accepted ADRs).
2. `agent/reports/phase_<N-1>/next_phase_handoff.md`.
3. The phase specification in `implementation_plan/`.
4. Existing tests for the affected modules.

Constraints:

- Modify only files explicitly allowed by the phase specification.
- Preserve all public interfaces unless the phase explicitly changes them.
- Do not introduce an additional framework when existing dependencies suffice.
- Do not download patient data.
- Do not place model weights in Git.
- Do not fabricate successful test results.
- Do not proceed past a failing acceptance condition.

Required completion:

- Implement the requested code.
- Add or update tests.
- Run all specified commands.
- Write the phase report (`agent/reports/phase_<N>/`).
- Record unresolved issues (or explicitly state none).
- Produce a next-phase handoff.
- Record any architectural decision discovered during implementation as a tracked ADR in `docs/architecture/` before closing the phase.
