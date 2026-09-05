# Terminal-Bench-Local

**Terminal-Bench-Local** provides versioned, computing-focused suites for
evaluating local models and quantizations. **Core-19 is the current benchmark
for new comparisons**. The superseded 20-task suite remains available only for
explicit reproduction of older runs.

## Core-19 task mix

Core-19 has 6 software-engineering tasks, 4 system-administration tasks, and 9
tasks spanning debugging, security, data/querying, and file operations.
Upstream metadata labels 3 easy, 13 medium, and 3 hard.

It removes only `build-pov-ray`, the clear serial timeout offender in the
finished data: 3/9 models passed it within two attempts, 8/16 attempts ended in
agent timeout, and its median model runtime was about 4.9 hours. Summed
historical per-task medians fall from roughly 14.8 hours to 9.9 hours (about
33%) while preserving short, medium, and long workflows. The two previously
considered removals, `break-filter-js-from-html` and
`llm-inference-batching-scheduler`, remain because 7/9 models passed each at
pass@2 and they preserve useful model-quality signal.

| Task | Area | Purpose |
| --- | --- | --- |
| `pypi-server` | software engineering | Build a Python package and serve it from a local package index. |
| `nginx-request-logging` | system administration | Configure Nginx logging, rate limits, and error pages. |
| `git-leak-recovery` | software engineering | Recover and then completely purge a leaked Git secret. |
| `fix-git` | software engineering | Recover detached commits and merge them into the main branch. |
| `cobol-modernization` | software engineering | Reimplement COBOL business logic in Python. |
| `regex-log` | data processing | Build a precise log and IPv4 matching expression. |
| `headless-terminal` | software engineering | Implement a persistent interactive terminal abstraction. |
| `mailman` | system administration | Configure Postfix and Mailman for a working mailing list. |
| `fix-ocaml-gc` | software engineering | Debug an OCaml garbage-collector crash in C. |
| `break-filter-js-from-html` | security | Construct HTML that exposes a sanitizer weakness. |
| `sqlite-with-gcov` | system administration | Build instrumented SQLite from vendored source. |
| `sparql-university` | data querying | Write a constrained aggregate SPARQL query. |
| `llm-inference-batching-scheduler` | ML systems | Implement and optimize a shape-aware batching scheduler. |
| `configure-git-webserver` | system administration | Deploy Git pushes automatically through Nginx. |
| `build-cython-ext` | debugging | Repair and compile Cython extensions against NumPy 2.x. |
| `extract-elf` | file operations | Parse an ELF binary and export memory values. |
| `openssl-selfsigned-cert` | security | Generate and validate a correctly configured TLS certificate. |
| `overfull-hbox` | debugging | Repair LaTeX layout under constrained edits. |
| `mteb-retrieve` | data science | Perform deterministic embedding-based retrieval. |

`mailman`, `fix-ocaml-gc`, `llm-inference-batching-scheduler`, and
`build-cython-ext` provide the sustained-work
portion of the suite without letting chronic timeouts dominate total runtime.

## Requirements

Install these host-side prerequisites:

- Linux and Python 3.11 or newer;
- one container stack that the current user can run without `sudo`:
  - Docker Engine with the Compose v2 `docker compose` command; or
  - Podman with a Docker-compatible `docker` command and a Compose provider
    such as `podman-compose`;
- [`uv`](https://docs.astral.sh/uv/) (recommended), or an existing Harbor
  0.20.0 installation that provides the `harbor` command.

`tmux` is optional but strongly recommended for full runs, which commonly last
many hours. It keeps the benchmark attached to a persistent terminal if the
user disconnects and lets a local coding agent inspect the same live output.

With `uv` or `uvx`, Harbor does not need to be installed manually. The runner
resolves the pinned `harbor==0.20.0` package when it is first needed and verifies
the CLI's exact version before a run. The first real run also pulls the task
container images, so allow outbound network access
and enough free disk space for Python packages and container images.

The OpenAI-compatible inference server must already be running and reachable
from this host. The repository does not install the model, inference engine,
GPU drivers, ROCm, CUDA, or Vulkan; those belong to the user's inference-server
setup. All built-in-suite task definitions are vendored locally, so no separate
Terminal-Bench checkout or project-specific Python environment is required.

## Suites and provenance

Runs select a versioned suite manifest with `--suite`. Two built-in manifests
are available:

- `core19` — the current timeout-pruned benchmark, retaining 19 tasks and
  removing only `build-pov-ray`;
- `legacy-mini20` — the superseded 20-task suite, retained only for explicit
  reproduction of older runs.

Suite manifests live under `suites/`. They record the exact upstream source,
revision, task content digest, tiers, and task root. The runner verifies every
digest before a run and stores the suite identity and per-task provenance with
the results. Results from different suite manifests are placed in separate
namespaces and are never merged by task filename alone.

`tasks/` contains the vendored Terminal-Bench 2.1 tasks. In `legacy-mini20`
1.0.1 and `core19` 1.0.0, `configure-git-webserver` has a verifier-only
hardening patch so invalid SSH setups fail promptly; its instruction and
environment are unchanged. Existing completed runs keep their original attempt
and reward data and are normalized to the Core-19 task denominator.
Use a built-in ID or a manifest path:

```bash
./terminal_bench.py list --suite core19
./terminal_bench.py doctor --suite core19 --tier smoke \
  --endpoint http://localhost:8080/v1
./terminal_bench.py run --suite core19 --tier full \
  --platform <platform> --model-name <canonical-model-name> \
  --engine <engine> --backend <compute-backend>
```

Core-19 is the no-argument default for current runs. Select `legacy-mini20`
explicitly—or use the old `--tasks-dir tasks` spelling—only for historical
reproducibility.

Check a fresh host with:

```bash
python3 --version
docker --version
docker compose version
uv --version  # Not needed when Harbor 0.20.0 is already installed.
./terminal_bench.py doctor --endpoint http://localhost:8080/v1
```

Pass the real endpoint, model, or context-length options to `doctor` when they
differ from the defaults. `doctor` queries `/models` and checks the local
runtime, Harbor launch path, and vendored tasks without starting inference.

## Run

### Agent-assisted setup

After cloning the repository, you can ask a coding agent:

> I want to run Terminal Bench.

The root `AGENTS.md` tells the agent to collect the run details in one concise
exchange, query the endpoint for its exact model ID and context capacity, run
the non-inference `doctor` checks, and then present a complete command and run
summary for confirmation. By default, the agent should give you the command and
recommend that you run it yourself in a terminal or persistent `tmux` session.
It may also offer to manage the run, but should do so only when asked and inside
`tmux` or another persistent session.

Expect to provide the desired scope, endpoint, hardware platform, inference
engine, compute backend, exact quantization or variant, and any special
inference profile. For example, a platform might be `strix-halo`, `gb10`, or
`r9700`, while a quantization might be `UD-Q4_K_XL`, `UD-IQ3_XXS`,
`Q4_0_ROCMI4`, or `MXFP4`. The runner normally discovers the served model ID
and context capacity itself.

### Commands

```bash
./terminal_bench.py doctor --suite core19
./terminal_bench.py run --suite core19 --tier full \
  --platform <platform> --model-name <canonical-model-name> \
  --engine <engine> --backend <compute-backend>
```

Core-19 is the default for new runs. Select `legacy-mini20` explicitly only to
reproduce a historical 20-task result. Every suite may define its own full and
smoke tiers. The default endpoint is `http://localhost:8080/v1`. A complete
Core-19 run might look like:

```bash
./terminal_bench.py run --suite core19 --tier full \
  --endpoint http://localhost:8080/v1 \
  --platform strix-halo \
  --platform-name "AMD Strix Halo" \
  --model-name Qwen3.8-27B \
  --engine llama.cpp \
  --backend rocm \
  --backend-version 10.0 \
  --quant UD-Q4_K_XL \
  --inference-profile mtp
```

To distribute a run across equivalent inference hosts, pass two or more URLs
with `--endpoints`:

```bash
./terminal_bench.py run --suite core19 --tier full \
  --endpoints http://host-a:8000/v1,http://host-b:8000/v1,http://host-c:8000/v1 \
  --model deepseek-v4-flash \
  --platform strix-halo \
  --model-name DeepSeek-V4-Vision_Exp \
  --engine DwarfStar \
  --backend rocm \
  --backend-version 10.0 \
  --quant IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8
```

Comma-separated and space-separated endpoint lists are both accepted.
`--endpoint` and `--endpoints` are mutually exclusive. Before launching work,
the runner checks that every URL advertises the same exact model ID, context
capacity, and stable model metadata. Hardware, engine, and backend identity
still come from the explicit run arguments because an OpenAI-compatible
endpoint cannot reliably report them.

Core-19 tasks are assigned with deterministic longest-processing-time
scheduling using historical mean attempt durations, including long-tail and
timeout cost. This balances estimated wall time across endpoints and places
expensive tasks first; suites without duration estimates fall back to
round-robin assignment. The runner starts one Harbor child job per non-empty
shard, so the default `--concurrency 1` means one task at a time on each
endpoint; a larger value applies per endpoint. Conditional retries use the same
scheduler, and normalized results record the endpoint used by each attempt. To
prevent concurrent Harbor progress renderers from corrupting the terminal,
each child writes to `jobs/<child-name>/harbor-console.log`; the
parent owns a fixed live dashboard with one row per endpoint, refreshed once per
second. The overall row shows total progress and the current pass rate across
graded tasks; each endpoint row shows its own progress, pass rate, active task
and elapsed time, pending count, and errors. Redirected or non-interactive
output instead prints a plain status block once per minute. On Ctrl+C, the
runner stops its Harbor children and removes containers whose Compose project
matches a trial in that interrupted campaign. Explicit `--keep-containers` runs
retain them as requested. Any trial result written with the exact
`CancelledError` interruption type is also discarded and returned to the
pending set; genuine failures and completed results are preserved.

Multi-endpoint runs create `jobs/<job-name>/orchestrator.json`, with child jobs
beside the parent named `<job-name>-endpoint1`, `<job-name>-endpoint2`, and so
on. Resume the parent to continue an interrupted distributed run:

```bash
./terminal_bench.py resume jobs/<job-name>
```

An interrupted single-endpoint job can instead be resumed on a replacement
endpoint or across additional equivalent endpoints. Stop the original command
with Ctrl+C and wait for it to exit, then run either:

```bash
./terminal_bench.py resume jobs/<job-name> \
  --endpoints http://host-a:8000/v1,http://host-b:8000/v1

./terminal_bench.py resume jobs/<job-name> \
  --endpoint http://replacement-host:8000/v1
```

The runner preserves every completed pass or genuine failure. A Ctrl+C
`CancelledError` artifact is discarded, as is any trial without a final
`result.json`; that task restarts from scratch without consuming an attempt.
The unfinished tasks are distributed using the normal weighted scheduler. The
added endpoints must advertise the stored model ID and matching model metadata
and context. This
changes only execution topology: model, platform, engine, backend, quant,
profile, suite, and attempt identity remain unchanged. After the current round,
normal conditional second attempts continue. An existing multi-endpoint
orchestrator cannot be repartitioned; resume its parent with the endpoints it
already records. During a redistributed resume, the overall dashboard retains
the full suite denominator and includes preserved results in both progress and
the live pass rate; endpoint rows remain scoped to their assigned unfinished
tasks.

### Job browser

Run the built-in job browser from the repository root:

```bash
./terminal_bench.py jobs
```

It lists campaigns newest-first and folds endpoint children and conditional
retry jobs into their parent campaign. Each entry shows start and finish or
last-activity time, completion and pass counts, suite/tier, readable model
name, quantization, inference profile, platform, engine/backend, endpoints, and
the raw job name. In an interactive terminal, select a number to resume an
incomplete campaign or rerun any campaign from scratch. Resume always displays
the stored endpoints and asks whether to change them. Live campaigns are
view-only, and every action that starts inference requires confirmation.

For a non-interactive inventory or scripts, use:

```bash
./terminal_bench.py jobs --list-only
./terminal_bench.py jobs --all --list-only
```

Every benchmark run must explicitly specify `--platform`, `--engine`, and
`--backend`. These values have no defaults because they are separate parts of
the recorded result identity:

- `--platform` is the hardware platform that ran the model, such as
  `strix-halo`; it is not the endpoint host. Do not use `localhost`, an IP
  address, or a port as a platform identifier.
- `--engine` is the inference server or implementation, such as `llama.cpp` or
  `DwarfStar`;
- `--backend` is the compute backend used by that engine, such as `rocm`,
  `vulkan`, `cuda`, `metal`, or `cpu`;
- `--engine-version` and `--backend-version` optionally preserve exact build,
  commit, API, runtime, or driver versions.

For example, a Vulkan llama.cpp deployment uses `--engine llama.cpp --backend
vulkan`. A ROCm DwarfStar deployment uses `--engine DwarfStar --backend rocm
--backend-version 7.14`. The deprecated `--rocm-version` option remains an
alias for `--backend-version` only when `--backend rocm` is selected.

`--platform-name` can optionally provide a human-readable platform name
alongside the required platform identifier. Engine and compute-backend identity
are included in job names, result-directory identity, cache matching, metadata,
and the results explorer, preventing Vulkan and ROCm attempts from colliding.

The runner discovers the exact served model ID and context capacity from
`/models`. Pass `--model` if the endpoint advertises more than one model. Every
run also requires `--model-name`: a stable human-readable family/revision such
as `DeepSeek-V4-Flash-0731` or `Qwen3.8-27B`. It must stay the same across
quants and inference profiles and must not include a GGUF extension, shard
number, quant, profile, or experimental tag. Pass `--context-length` only when
an explicit user-requested override is required.

Authentication is not normally needed for a local endpoint, so omit
`--api-key` by default. If querying `/models` returns `401` or `403`, the
endpoint requires authentication; set `TBENCH_API_KEY` or pass `--api-key` for
that endpoint.

Record variant identity in three separate fields:

- `--quant` is only the numeric format or quantization, such as `UD-IQ3_XXS`,
  `Q4_0_ROCMI4`, or `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8`;
- `--inference-profile` is only the serving profile, such as `mtp16` or
  `DSpark`; and
- `--tag` is an optional remaining label, such as `chat-v2-imatrix-0731`.

Do not combine these fields or pass a complete model filename as any of them.
They are stored separately, included in run identity, and exposed as separate
results-explorer filters. Legacy exported metadata containing `model_tag` is
read as quantization during migration, but new commands and exports do not use
that field.

Single-endpoint runs are sequential by default. Multi-endpoint runs explicitly
opt into parallel execution across the supplied hosts while remaining
sequential within each host by default. Runs allow up to two attempts per task;
attempt two runs only when attempt one fails. Terminus-2 summarizes context when
it approaches the advertised limit. Each attempt has a three-hour agent
timeout; there is no model-call, turn, or output-token cap. Results report the
aggregate pass rate for the configured attempt budget: the default is pass@2,
while `--attempts 1` produces pass@1.

### Running in tmux

A full run is best started by the user in a persistent terminal:

```bash
tmux new-session -s terminal-bench-mini
# Paste the generated ./terminal_bench.py run ... command.
```

Detach without stopping the benchmark by pressing `Ctrl-b`, then `d`. Reattach
later with:

```bash
tmux attach-session -t terminal-bench-mini
```

If a coding agent starts the benchmark for you, it should report the tmux
session name and generated `jobs/<job-name>` directory. Those give the agent
both the live terminal output and durable result artifacts when you later ask
for a status update.

Interrupted jobs and existing failures can be continued without rerunning
successful tasks. Pass the same custom manifest again when retrying a result
from a non-built-in suite:

```bash
./terminal_bench.py resume jobs/<job-name>
./terminal_bench.py resume jobs/<single-job-name> \
  --endpoints http://host-a:8000/v1,http://host-b:8000/v1
./terminal_bench.py retry-failed results/suites/<suite-hash>/<platform>/<model>_results
./terminal_bench.py jobs
```

## Results

Raw Harbor jobs are written to `jobs/`. Manifest-backed results are written to
`results/suites/<suite-id>-<manifest-hash>/<platform>/<model>_results/`, with
one result and copied ATIF transcript per task. Older unscoped result paths
remain readable. Quantization strings are retained in model and directory
names. The exact inference engine and compute backend are recorded separately.

When Harbor's aggregate counters confirm that a run or resumed job is terminal,
the runner automatically exports the normalized task results, writes the
aggregate `summary.json`, and rebuilds the relevant suite index. The presence
of a live `result.json` alone does not mark a job complete. No separate
result-generation command is normally required. Use these commands to inspect
a raw job or rebuild indexes from existing exported results:

```bash
./terminal_bench.py summary jobs/<job-name>
./terminal_bench.py results
```

The bundled completed runs are exposed on the same 19-task Core-19 denominator;
the removed `build-pov-ray` result does not appear in the explorer or its
aggregates.

### Results explorer

The static web UI in `docs/` provides searchable model cards, aggregate pass@N
results and per-attempt breakdowns, a cross-model task matrix, task metadata,
timings, token usage, failure classifications, verifier excerpts, run profiles,
and links to committed results and transcripts.

After a run exports new results, regenerate the explorer's compact dataset and
serve only the generated documentation directory:

```bash
python3 docs/build_data.py
python3 -m http.server --bind 127.0.0.1 --directory docs 8000
```

Open `http://127.0.0.1:8000/` to preview it. The site is dependency-free
and requires no Node.js, npm, or frontend build. Run `docs/build_data.py` again
after each new export; the running HTTP server will then serve the updated
Core-19 `docs/data.json`. Local `jobs/` directories do not need to be committed: when
present, the data builder extracts short verifier evidence into
`docs/data.json`, while links target normalized artifacts under `results/` and
task definitions under `tasks/`. Publishing or committing results is separate
from local visualization and should be done only when intended.

## Layout

```text
terminal_bench.py   runner and CLI
suite_manifest.py   suite validation and deterministic task identity
suites/             built-in suite manifests and tiers
results.py          stable result export and indexing
tasks/              vendored Terminal-Bench 2.1 task pool
subsets/            legacy mini-20 task lists
compat/podman/      Docker/Podman compatibility shim
results/            suite-scoped results and bundled legacy baselines
docs/               static results explorer and generated site dataset
tests/              runner and result-store tests
```

Run the tests with:

```bash
python3 -m unittest discover -s tests -v
```

The legacy task definitions come from
[`harbor-framework/terminal-bench-2-1`](https://github.com/harbor-framework/terminal-bench-2-1)
at revision `5c8eadf1f393183288fa08b8f73ca9a469cc5e00` and retain their original
authors and Apache-2.0 license metadata.
