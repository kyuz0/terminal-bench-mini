# Agent guide

## Scope

This repository is **Terminal-Bench-Mini-20**: a small, self-contained
Terminal-Bench 2.1 derivative for comparing locally served model variants and
quantizations. It tests medium- to long-horizon agentic workflows in which a
model uses terminal tools, writes and debugs code, configures systems, and
completes multi-step tasks.

## User communication

Keep every user-facing response direct and brief.

- Prefer short bullet points or a compact table.
- Lead with the command, result, or current status.
- Ask all required questions together; do not drip-feed them.
- Do not narrate routine actions, repeat known context, or add unrequested
  background.
- Explain only what the user needs to choose, run, fix, or understand next.

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
- the canonical human-readable model family and revision for `--model-name`.
  Keep it identical across quants and inference profiles, and exclude file
  extensions, shard numbers, quantization, inference profiles, and experimental
  tags. For example, every applicable variant is `DeepSeek-V4-Flash-0731`;
- the exact numeric format or quantization for `--quant`, such as
  `UD-Q4_K_XL`, `UD-IQ3_XXS`, `Q4_0_ROCMI4`, `MXFP4`, or
  `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8`. Never put a complete model filename,
  model-family prefix, shard suffix, inference profile, or free-form tag here;
- any special inference profile for `--inference-profile`, such as `mtp`,
  `mtp16`, or `DSpark`, separately from quantization; and
- an optional `--tag` only for remaining variant information such as
  `chat-v2-imatrix-0731`. Never use `--tag` for quantization or inference
  profile. Omit it when no additional distinction is needed.

Use the endpoint's `/models` response to discover the served model ID and
context capacity. The runner discovers and records the endpoint-advertised
maximum automatically: never include `--context-length` in a CLI command
unless the user explicitly asks for an override. `--model` must be the exact
advertised ID, even when it is a full model-file path. If the endpoint
advertises multiple models, show the available IDs and ask the user to select
one. Warn the user when the advertised maximum context is below 262,144 tokens.

Do not begin by asking about `TBENCH_API_KEY`, and omit `--api-key` from normal
local commands. The runner's default placeholder is suitable for
unauthenticated local endpoints. If the `/models` query specifically returns
`401` or `403`, explain that the endpoint requires authentication and ask the
user to configure `TBENCH_API_KEY` or provide `--api-key`. Do not echo or repeat
a secret supplied through the environment.

Once the endpoint details are resolved, run `./terminal_bench.py doctor` with
the applicable `--endpoint` and `--model` arguments. This validates the
endpoint, model metadata, container runtime, Harbor availability, and vendored
tasks without running inference. Resolve any failed prerequisite before
proposing a real run.

Unless the user explicitly overrides them, preserve the benchmark defaults:

- concurrency `1`;
- up to `2` attempts per task, with the second attempt only after failure;
- a three-hour agent timeout per attempt;
- automatic container cleanup after each task.

Do not include default-valued `--agent-timeout` in a command. Include it only
when the user explicitly chooses a non-default timeout.

Before execution, show the user:

1. a compact summary of the tier and task count, endpoint, exact model ID,
   context length, canonical model name, hardware platform, engine and version,
   backend and version, quantization, inference profile, optional tag, attempts,
   concurrency, timeout, and conditional-retry behavior;
2. the complete copy-pasteable `./terminal_bench.py run ...` command, with the
   tier and identity fields explicit; and
3. a reminder that a full run can take many hours and that settings can still
   be corrected or clarified.

The default handoff is to give the user the command and recommend that they run
it themselves in a terminal, preferably inside `tmux`. Include concise tmux
instructions when useful. Ensure every shell continuation backslash is the last
character on its line so the command is genuinely copy-pasteable.

Offer to run and monitor the command for the user, but do not treat approval of
the command's settings as approval for agent-managed execution. Start it only
when the user explicitly asks the agent to run it. An agent-managed full run
must use a persistent `tmux` session or an equivalent persistent supervisor,
never a transient tool shell. Run it from the repository root, report the
session name, capture the generated job name and `jobs/<job-name>` path, and
verify that Harbor actually started. If tmux is unavailable, explain the
alternative before launching.

## Running and status updates

When the user asks for an update, inspect both the live session and durable job
artifacts. For a tmux-managed run, use `tmux list-sessions` and capture the
relevant pane. Identify the exact job from the startup output or
`jobs/<job-name>/runner-meta.json`, then inspect its `result.json` and task-local
artifacts. Do not rely on the progress bar alone.

Interpret Harbor output carefully:

- `11/20 Mean: 1.000` means 11 trials have completed and the mean reward among
  the completed trials is currently 1.0. It is not the final 20-task score;
- a line such as `litellm.Timeout ... timeout value=600.0` records a timed-out
  model request. It is not automatically a failed task or stopped benchmark;
  if later progress appears, the overall run continued. Use the task result and
  exception artifacts before classifying the outcome;
- `1:35:27 build-pov-ray... running agent` is the elapsed time for the current
  task. Sparse output during a long task is not proof that it is stuck; and
- `-:--:--` means Harbor has no usable ETA.

A useful status update states:

- whether the tmux session, Harbor process, and relevant task container are
  active;
- the current attempt round and exact job directory;
- completed tasks versus total tasks, the current task, and its elapsed time;
- confirmed passes among completed tasks, clearly labelled as provisional until
  the configured attempt budget finishes;
- observed timeouts or errors and whether the run subsequently progressed; and
- whether a conditional second-attempt job is pending, running, or complete.

Use coarse checks rather than continuous polling. Do not call a benchmark
finished merely because attempt one ended: with the default policy, Harbor may
move into a separate `-attempt2` job for failed tasks. Treat the run as complete
only after the configured attempt policy has finished and the normalized result
set has been exported.
on ocm 10.0
## Results and visualization

A completed run or resumed Harbor job automatically exports normalized results
under `results/<platform>/<model>_results/`, writes that result set's
`summary.json`, and rebuilds `results/index.json`. Do not hand-edit any of these
artifacts. `./terminal_bench.py results` is available to rebuild the aggregate
index from already exported result sets; it is not normally needed after a
successful automatic export.

At the end of a run, give the user a compact result summary containing:

- the exact platform, canonical model name, served model ID, quantization,
  inference profile, optional tag, engine, backend, and configured attempt budget;
- the aggregate passed/total count and pass@N rate for that attempt budget;
- the result directory and raw Harbor job directory;
- failed or errored tasks, clearly distinguishing verifier, agent-timeout, and
  harness or endpoint errors when the artifacts support that distinction; and
- timing and token totals when available.

Do not present an interrupted or partially exported job as a final aggregate.
Use `./terminal_bench.py summary jobs/<job-name>` for raw-job status and
`./terminal_bench.py resume jobs/<job-name>` when the user wants to continue it.

To generate the local results explorer after an export, run from the repository
root:

```bash
python3 docs/build_data.py
python3 -m http.server 8000
```

Tell the user to open `http://localhost:8000/docs/`. The viewer is static and
has no Node.js or npm dependency. Re-run `python3 docs/build_data.py` after each
new result export; an already-running HTTP server will serve the regenerated
dataset. If port 8000 is occupied, select another port and report the matching
URL. Do not publish, commit, or push results unless the user asks.

## Validation

Run commands from this repository's root:

```bash
python3 -m py_compile terminal_bench.py results.py compat/podman/docker
python3 -m unittest discover -s tests -v
./terminal_bench.py list
./terminal_bench.py results
```

To validate configuration and container discovery without running inference,
using a reachable endpoint that advertises the selected model:

```bash
./terminal_bench.py run --tier smoke \
  --endpoint <endpoint> \
  --model <advertised-model-id> \
  --platform test-local \
  --model-name test-model \
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
  --platform <platform> --model-name <canonical-model-name> \
  --engine <engine> --backend <compute-backend> [model options]

# Continue an interrupted Harbor job
./terminal_bench.py resume jobs/<job-name>

# Give failed tasks their next conditional attempt
./terminal_bench.py retry-failed results/<platform>/<model>_results
```
