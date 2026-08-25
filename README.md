# Terminal-Bench-Local

Terminal-Bench-Local is a 20-task, computing-focused subset of Terminal-Bench
2.1 for evaluating local models and quantizations. It checks whether a model
can remain coherent, use a terminal, write and debug code, configure services,
and finish multi-step work. It is not an official Terminal-Bench 2.1 score.

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

- Linux with Docker Compose, or Podman with a Docker-compatible `docker` CLI;
- Python 3.11 or newer;
- [`uv`](https://docs.astral.sh/uv/) or an installed `harbor` command;
- one OpenAI-compatible model endpoint.

The first run downloads Harbor 0.20.0 and task container images. The 20 task
definitions are vendored in `tasks/`; no separate Terminal-Bench checkout is
required.

## Run

```bash
./terminal_bench.py doctor
./terminal_bench.py run --tier smoke \
  --platform <platform> --engine <engine> --backend <compute-backend>
./terminal_bench.py run --tier full \
  --platform <platform> --engine <engine> --backend <compute-backend>
```

The default endpoint is `http://localhost:8080/v1`. A complete local run might
look like:

```bash
./terminal_bench.py run --tier full \
  --endpoint http://localhost:8080/v1 \
  --platform strix-halo \
  --platform-name "AMD Strix Halo" \
  --engine llama.cpp \
  --backend rocm \
  --backend-version 7.14 \
  --model-tag mtp \
  --inference-profile mtp
```

Every benchmark run must explicitly specify `--platform`, `--engine`, and
`--backend`. These values have no defaults because they are separate parts of
the recorded result identity:

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

The runner discovers the model ID and context capacity from `/models`. Pass
`--model` if the endpoint advertises more than one model, or `--context-length`
if it does not advertise its capacity.

When an endpoint exposes the same model ID for different quantizations or
variants, pass the exact variant name with `--model-tag`. The full tag is
recorded in result metadata and is included in both the Harbor job-directory
name and the exported result-directory identity. Automatic attempt job names
retain the same tag.

Runs are sequential and allow up to two attempts per task. Attempt two runs
only when attempt one fails. Terminus-2 summarizes context when it approaches
the advertised limit. Each attempt has a three-hour agent timeout; there is no
model-call, turn, or output-token cap. Use `--attempts 1` for strict pass@1.

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

The static web UI in `docs/` provides searchable model cards, pass@1 and
pass-within-attempt comparisons, a cross-model task matrix, task metadata,
timings, token usage, failure classifications, verifier excerpts, run profiles,
and links to committed results and transcripts.

Regenerate its compact dataset after exporting or committing new results:

```bash
python3 docs/build_data.py
python3 -m http.server 8000
```

Open `http://localhost:8000/docs/` to preview it. The site is dependency-free
and can be published directly with GitHub Pages using `docs/` as the source.
Local `jobs/` directories do not need to be committed: when present, the data
builder extracts short verifier evidence into `docs/data.json`, while public
links target normalized artifacts under `results/` and task definitions under
`tasks/`.

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
