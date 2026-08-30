# Agent guide

## Scope

This repository is **Terminal-Bench-Mini-20**: a small, self-contained
Terminal-Bench 2.1 derivative for comparing locally served model variants and
quantizations. It tests medium- to long-horizon agentic workflows in which a
model uses terminal tools, writes and debugs code, configures systems, and
completes multi-step tasks.

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
- Report the aggregate pass rate across the selected task set using the attempt
  budget chosen for that run. The default two-attempt run is reported as
  pass@2; a run with `--attempts 1` is reported as pass@1.

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

## Fresh-clone prerequisites

Before constructing a first run, verify the host environment and explain any
missing prerequisite together rather than discovering dependencies during a
long benchmark. The user needs:

- Linux with Python 3.11 or newer;
- either Docker Engine with Compose v2, or Podman with a Docker-compatible
  `docker` command and a Compose provider such as `podman-compose`;
- permission to run the selected container stack as the current user;
- `uv` or `uvx`, unless Harbor 0.20.0 is already installed as `harbor`; and
- an OpenAI-compatible inference endpoint that is already running and
  host-reachable.

Prefer the `uv` path on a fresh machine. Do not tell the user to install Harbor
separately when `uv` or `uvx` is available: the runner resolves the pinned
`harbor==0.20.0` package. A preinstalled `harbor` command must be version 0.20.0.
The first real run needs outbound network access and disk space to download
Harbor when necessary and pull task container images.

The benchmark repository does not install or configure the model server, GPU
drivers, ROCm, CUDA, Vulkan, or the selected inference engine. Those describe
the external server and are recorded as run identity. Do not propose installing
hardware-specific inference dependencies merely to run the benchmark harness.

Check `python3 --version`, `docker --version`, `docker compose version`, and
`uv --version` or `harbor --version` as applicable. If something is missing,
report exactly what is absent and ask before installing system packages.

## Starting a benchmark for a user

When a user says something broad such as "I want to run Terminal Bench," help
them construct the command before starting the benchmark. Do not drip-feed
questions one at a time. Ask for the information that cannot be discovered in
one concise exchange, query the endpoint for the rest, and then present the
resolved configuration and command together for confirmation.

Collect or confirm:

- the scope: the full 20-task tier is the default. Use the one-task smoke tier
  only when the user explicitly asks for a quick validation, or select one
  explicit task when requested;
- the host-visible OpenAI-compatible endpoint, for example
  `http://localhost:8080/v1`;
- the hardware platform identifier, such as `strix-halo`, `gb10`, or `r9700`,
  plus an optional human-readable name. The platform is the hardware identity,
  never `localhost`, an IP address, or a port;
- the inference engine, such as `llama.cpp`, `DwarfStar`, or `vLLM`, and its
  version or commit when known;
- the compute backend, such as `rocm-10.0`, `vulkan`, `cuda`, `metal`, or `cpu`, and
  its version when known;
- the exact quantization or variant label that should identify the run. Pass it
  as `--model-tag`; examples include `UD-Q4_K_XL`, `UD-IQ3_XXS`,
  `Q4_0_ROCMI4`, `MXFP4`, and longer custom quant names;
- any special inference profile, such as MTP/DSpark or `mtp16/dspark6`, separately from the
  quantization label.

Use the endpoint's `/models` response to discover the served model ID and
context capacity. `--model` must be the exact advertised ID, even when it is a
full model-file path. If the endpoint advertises multiple models, show the
available IDs and ask the user to select one. Ask for `--context-length` only
when the selected model's metadata does not advertise it.

Do not begin by asking about `TBENCH_API_KEY`, and omit `--api-key` from normal
local commands. The runner's default placeholder is suitable for
unauthenticated local endpoints. If the `/models` query specifically returns
`401` or `403`, explain that the endpoint requires authentication and ask the
user to configure `TBENCH_API_KEY` or provide `--api-key`. Do not echo or repeat
a secret supplied through the environment.

Once the endpoint details are resolved, run `./terminal_bench.py doctor` with
the applicable `--endpoint`, `--model`, and `--context-length` arguments. This
validates the endpoint, model metadata, container runtime, Harbor availability,
and vendored tasks without running inference. Resolve any failed prerequisite
before proposing a real run.

Unless the user explicitly overrides them, preserve the benchmark defaults:

- concurrency `1`;
- up to `2` attempts per task, with the second attempt only after failure;
- a three-hour agent timeout per attempt;
- automatic container cleanup after each task.

Before execution, show the user:

1. a compact summary of the tier and task count, endpoint, exact model ID,
   context length, hardware platform, engine and version, backend and version,
   quantization/model tag, inference profile, attempts, concurrency, timeout,
   and conditional-retry behavior;
2. the complete copy-pasteable `./terminal_bench.py run ...` command, with the
   tier and identity fields explicit; and
3. a reminder that a full run can take many hours and that settings can still
   be corrected or clarified.

Do not start a real benchmark until the user confirms the summarized command.

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
