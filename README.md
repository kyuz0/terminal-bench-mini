# Terminal-Bench-Mini-20

**Terminal-Bench-Mini-20** is a 20-task, computing-focused subset of
Terminal-Bench 2.1 for evaluating local models and quantizations. It checks
whether a model can remain coherent, use a terminal, write and debug code,
configure services, and finish multi-step work.

## Task mix

The subset has 7 software-engineering tasks, 4 system-administration tasks,
and 9 tasks spanning debugging, security, data/querying, ML scheduling, and
file operations. Upstream metadata labels 3 easy, 13 medium, and 4 hard.

Selection used published results from six strong model configurations. Ten
tasks had high pass rates (87–100%) and ten had medium pass rates (53–78%).
Rarely solved floor tasks and non-computing work such as protein modelling were
excluded. A few longer tasks remain to test sustained work and context handling.

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
| `build-pov-ray` | software engineering | Port and compile legacy POV-Ray source. |
| `openssl-selfsigned-cert` | security | Generate and validate a correctly configured TLS certificate. |
| `overfull-hbox` | debugging | Repair LaTeX layout under constrained edits. |
| `mteb-retrieve` | data science | Perform deterministic embedding-based retrieval. |

`mailman`, `fix-ocaml-gc`, `llm-inference-batching-scheduler`,
`build-cython-ext`, and `build-pov-ray` provide the longer-horizon portion of
the suite. The remaining tasks keep the benchmark practical on local hardware.

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
resolves the pinned `harbor==0.20.0` package when it is first needed. The first
real run also pulls the task container images, so allow outbound network access
and enough free disk space for Python packages and container images.

The OpenAI-compatible inference server must already be running and reachable
from this host. The repository does not install the model, inference engine,
GPU drivers, ROCm, CUDA, or Vulkan; those belong to the user's inference-server
setup. The 20 task definitions are already vendored in `tasks/`, so no separate
Terminal-Bench checkout or project-specific Python environment is required.

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
./terminal_bench.py doctor
./terminal_bench.py run --tier full \
  --platform <platform> --model-name <canonical-model-name> \
  --engine <engine> --backend <compute-backend>
./terminal_bench.py run --tier smoke \
  --platform <platform> --model-name <canonical-model-name> \
  --engine <engine> --backend <compute-backend>
```

The full 20-task tier is the default. The smoke tier exists only as an explicit
quick validation. The default endpoint is `http://localhost:8080/v1`. A
complete local run might look like:

```bash
./terminal_bench.py run --tier full \
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

Runs are sequential and allow up to two attempts per task. Attempt two runs
only when attempt one fails. Terminus-2 summarizes context when it approaches
the advertised limit. Each attempt has a three-hour agent timeout; there is no
model-call, turn, or output-token cap. Results report the aggregate pass rate
for the configured attempt budget: the default is pass@2, while
`--attempts 1` produces pass@1.

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
successful tasks:

```bash
./terminal_bench.py resume jobs/<job-name>
./terminal_bench.py retry-failed results/<platform>/<model>_results
```

## Results

Raw Harbor jobs are written to `jobs/`. Stable results are written to
`results/<platform>/<model>_results/`, with one result and copied ATIF
transcript per task. Quantization strings are retained in model and directory
names. The exact inference engine and compute backend are recorded separately.

When a run or resumed job produces a Harbor `result.json`, the runner
automatically exports the normalized task results, writes the aggregate
`summary.json`, and rebuilds `results/index.json`. No separate result-generation
command is normally required. Use these commands to inspect a raw job or
rebuild the aggregate index from existing exported results:

```bash
./terminal_bench.py summary jobs/<job-name>
./terminal_bench.py results
```

The bundled preliminary baseline is:

```text
Qwen3.6-35B-A3B-UD-Q4_K_XL + MTP + Terminus-2
AMD Strix Halo, llama.cpp, 262,144-token context
11/20 pass@1 (55%)
```

All eleven passes occurred on attempt one. `build-pov-ray` also has a failed
second attempt preserved in its result file; therefore the bundled data should
be reported as pass@1, not as a complete pass-within-two run. The baseline is
included to demonstrate the format and provide a first quantized local-model
reference point.

### Results explorer

The static web UI in `docs/` provides searchable model cards, aggregate pass@N
results and per-attempt breakdowns, a cross-model task matrix, task metadata,
timings, token usage, failure classifications, verifier excerpts, run profiles,
and links to committed results and transcripts.

After a run exports new results, regenerate the explorer's compact dataset and
serve the repository root:

```bash
python3 docs/build_data.py
python3 -m http.server 8000
```

Open `http://localhost:8000/docs/` to preview it. The site is dependency-free
and requires no Node.js, npm, or frontend build. Run `docs/build_data.py` again
after each new export; the running HTTP server will then serve the updated
`docs/data.json`. Local `jobs/` directories do not need to be committed: when
present, the data builder extracts short verifier evidence into
`docs/data.json`, while links target normalized artifacts under `results/` and
task definitions under `tasks/`. Publishing or committing results is separate
from local visualization and should be done only when intended.

## Layout

```text
terminal_bench.py   runner and CLI
results.py          stable result export and indexing
tasks/              vendored 20-task dataset
subsets/            full and smoke task lists
compat/podman/      Docker/Podman compatibility shim
results/            bundled Qwen baseline and transcripts
docs/               static results explorer and generated site dataset
tests/              runner and result-store tests
```

Run the tests with:

```bash
python3 -m unittest discover -s tests -v
```

Task definitions come from
[`harbor-framework/terminal-bench-2-1`](https://github.com/harbor-framework/terminal-bench-2-1)
at revision `5c8eadf1f393183288fa08b8f73ca9a469cc5e00` and retain their original
authors and Apache-2.0 license metadata.
