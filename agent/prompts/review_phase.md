# Review Phase Prompt

You are reviewing the implementation of Phase <N> of the medical foundation-model framework.

Read:

1. The phase specification in `implementation_plan/`.
2. The phase report in `agent/reports/phase_<N>/`.
3. The diff of every file in `files_changed.txt`.

Check, in order:

1. **Scope:** every changed file is inside the phase's allowed areas; nothing forbidden was touched.
2. **Contract conformance:** public interfaces match Phase 02 contracts and accepted ADRs; no untracked architectural decision was made (if one was, an ADR must exist).
3. **Governance:** no patient data, no weights in Git; license/accelerator-status registry rules intact; PHI fail-closed behavior preserved.
4. **Tests:** new behavior is covered; no silently skipped tests; no fabricated results — re-run the smoke command yourself.
5. **Quality:** code matches project conventions; no dead code, no speculative generality.

Output:

- A verdict per checklist section: `pass` or `fail` with file:line evidence.
- A list of required fixes, ordered by severity.
- Approve only when every section passes and the acceptance command exits 0.
