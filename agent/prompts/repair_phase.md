# Repair Phase Prompt

You are repairing Phase <N> of the medical foundation-model framework after a failed test or review.

Input:

1. The failing test names and output, or the review's required-fix list.
2. The phase specification and scope boundaries.
3. The current phase report.

Rules:

- Fix the root cause, not the test, unless the test is demonstrably wrong (then justify in the phase report).
- Stay within the phase's allowed files. If the fix requires touching a forbidden file, stop and escalate: record it in `unresolved_issues.md` and the handoff.
- Do not weaken an assertion, delete a test, or mark a criterion `not_applicable` to make the gate pass.
- If the failure reveals an architectural decision, write a tracked ADR before proceeding.

Completion:

- Re-run the smoke and acceptance commands.
- Update `test_results.json` and `acceptance.json` with the new evidence.
- Append a short repair note to `summary.md` describing cause and fix.
