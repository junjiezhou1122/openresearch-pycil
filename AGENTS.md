# AGENTS.md

- Fail fast. Never append `|| true`, `; true`, or equivalent error suppression to experiment commands.
- Fix root causes and preserve stdout, stderr, exit code, commit SHA, and runtime identity for every run.
- The default worker edit surface is: `algorithm.py`.
- Do not modify hidden-test data, evaluator code, result ledgers, acceptance verdicts, or Git tags.
- A validation improvement is a proxy observation, not proof of holdout success.
- Update TASK.md or BASELINE.md whenever the task contract or baseline changes.
- Create a branch before broad refactors or experimental changes.
