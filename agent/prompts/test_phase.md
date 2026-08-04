# Test Phase Prompt

You are validating Phase <N> of the medical foundation-model framework.

Steps:

1. Run the phase smoke command exactly as written in the phase specification.
2. Run the phase acceptance command: `python -m medfm.tools.validate_phase --phase <N>`.
3. Run the full test suite for affected modules: `python -m pytest tests/ -q` (or the phase-scoped subset where specified).
4. Confirm: no failures, no skips without a recorded reason, no warnings treated as acceptable without justification.

Rules:

- Report actual command output. Never fabricate a pass.
- If a test fails, stop: hand off to the repair prompt with the failing test names and output.
- If a test is missing for required behavior, that is a failure — report it.
- Record results in `agent/reports/phase_<N>/test_results.json` with per-test status, duration, and the exact commands run.
