# Agent guide

## Scope

This repository is **Terminal-Bench-Local-20**: a small, self-contained
Terminal-Bench 2.1 derivative for comparing locally served model variants and
quantizations. It is not the full benchmark and its scores must not be reported
as official Terminal-Bench 2.1 scores.

Keep this project independent. Do not add dependencies on the parent DeepSWE
workspace, Pier, its research clones, or paths outside this repository.

## Benchmark invariants

- `subsets/full.txt` is the canonical 20-task set; `subsets/smoke.txt` is the
  quick validation set.
- `tasks/` contains vendored upstream task definitions at the revision recorded
  in `tasks/REVISION`. Editing a task changes the benchmark.
- Runs use Harbor 0.20.0 with Terminus-2 2.0.0 against one OpenAI-compatible
  local endpoint.
- Run one task at a time by default. Local inference servers should not receive
  concurrent benchmark requests unless the user explicitly opts in.
- The default is up to two attempts per task, with the second attempt run only
  when the first fails.
- Each attempt has a three-hour agent timeout. Do not add arbitrary model-call,
  turn, or output-token limits.
- Preserve the exact model identifier, quantization suffix, inference engine,
  compute backend and their versions, context size, platform, and inference
  profile in results.
- A task passes only when its final `reward` is exactly `1`. Partial verifier
  metrics are not passes.
- Report the checked-in baseline as pass@1 unless explicitly reporting the
  separate pass-within-two result.

## Repository map

- `terminal_bench.py` — runner, resume/retry commands, endpoint discovery, and
  Harbor configuration.
- `results.py` — normalized result export, attempt merging, and aggregate index.
- `compat/podman/docker` — Docker-compatible shim used when Harbor runs through
  Podman.
- `subsets/` — benchmark task lists.
- `tasks/` — vendored benchmark fixtures; do not casually modify them.
- `results/` — checked-in Qwen baseline and transcripts.
- `jobs/` and `.runner/` — ignored runtime state and generated configuration.

Do not hand-edit scores or transcripts in `results/`. Export them through the
runner so attempt history and model identity remain consistent. Do not expose
task `solution/` content to an evaluated agent.

If the task set changes, update `subsets/full.txt`, the task summary and score
denominator in `README.md`, and any tests that assert the set size. Preserve the
upstream license and provenance in `NOTICE.md`.

Keep exploratory notes out of `README.md`. Update user documentation only when
the repository's actual behavior, task set, or methodology changes.

## Validation

Run commands from this repository's root:

```bash
python3 -m py_compile terminal_bench.py results.py compat/podman/docker
python3 -m unittest discover -s tests -v
./terminal_bench.py list
./terminal_bench.py results
```

To validate configuration and container discovery without running inference:

```bash
./terminal_bench.py run --tier smoke \
  --skip-endpoint-check \
  --model test/local-model-Q4_K_M.gguf \
  --context-length 262144 \
  --platform test-local \
  --engine llama.cpp \
  --backend rocm \
  --backend-version test \
  --dry-run
```

Do not start a real benchmark run unless the user asks for it: even the reduced
suite can consume many hours of local inference.

## Common operations

```bash
# Run the full 20-task set
./terminal_bench.py run --tier full \
  --platform <platform> --engine <engine> --backend <compute-backend> [model options]

# Continue an interrupted Harbor job
./terminal_bench.py resume jobs/<job-name>

# Give failed tasks their next conditional attempt
./terminal_bench.py retry-failed results/<platform>/<model>_results
```
