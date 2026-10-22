---
trigger: always_on
---

## Safety and Change-Control Rules

1. Inspect relevant files before editing them.
2. Make small, task-specific changes. Never rewrite the whole project to fix one issue.
3. Never delete, overwrite, rename, or move raw datasets, project reports, Git history, or user-created files.
4. Never run destructive commands such as `git reset --hard`, `git clean -fd`, recursive deletion, or bulk file replacement.
5. Never modify files outside the project folder unless explicitly instructed.
6. Never expose, upload, or commit passwords, API keys, tokens, credentials, or environment files containing secrets.
7. Install dependencies only inside `.venv`. Check existing dependencies before installing or upgrading packages.
8. Do not change global Python settings, execution policies, system configuration, or unrelated software.
9. Before major changes, inspect Git status and preserve existing user modifications.
10. After each task, run focused tests and verify that the application still works.
11. If a test fails, investigate the actual error. Do not hide it, fabricate success, or remove tests to make them pass.
12. Do not automatically commit changes. Report the changed files and let the user review before committing.
13. If a change could cause data loss, expose credentials, affect files outside the project, or alter system security, stop and ask for approval.
14. Keep responses concise: report changes, tests, failures, and the next action in at most five bullets.