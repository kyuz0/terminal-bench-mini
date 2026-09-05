# Provenance

The task definitions under `tasks/` are selected Terminal-Bench 2.1 tasks from
`harbor-framework/terminal-bench-2-1`, revision
`5c8eadf1f393183288fa08b8f73ca9a469cc5e00`. The verifier for
`configure-git-webserver` is locally hardened so invalid SSH configurations
fail promptly instead of waiting for the outer verifier timeout; its
agent-facing instruction and environment are unchanged. All other task content
is unmodified. Task authorship is retained in each `task.toml`; the upstream
Apache-2.0 license is included as `LICENSE`.

The result set under `results/` was produced locally with the model, endpoint,
agent, hardware, timing, and token metadata recorded in its JSON files. Raw
Harbor scratch jobs are not bundled; copied ATIF transcripts and normalized
per-task results are included.
