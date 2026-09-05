# Results explorer

`docs/` is a dependency-free static results explorer intended for GitHub Pages.
It reads `data.json` in the browser and does not require an API, database, or
application server.

Regenerate the current Core-19 dataset whenever committed benchmark results
change:

```bash
python3 docs/build_data.py
```

The no-argument form selects Core-19 and its 19-task denominator. Finished
Mini-20 results have been normalized to the 19 retained tasks so every model in
the viewer uses the same Core-19 denominator.

Core-19 removes only `build-pov-ray`, which passed for 3/9 finished models,
produced 8 agent-timeout attempts, and had a median model runtime near 4.9
hours. It keeps the other 19 tasks—including the two 7/9 pass@2 tasks previously
considered for removal—to preserve model-quality signal. The summed historical
median runtime estimate falls from about 14.8 to 9.9 hours.

Use `--suite legacy-mini20` only when explicitly reproducing the superseded
20-task benchmark.

Preview it locally from the repository root:

```bash
python3 -m http.server --bind 127.0.0.1 --directory docs 8000
```

Then open `http://127.0.0.1:8000/`. Opening `index.html` directly through
`file://` will not work because browsers block the JSON fetch.

The builder reads:

- normalized run and attempt data from the selected namespace under `results/`;
- task descriptions, metadata, and evaluated instructions from the selected
  suite's task roots;
- short verifier failure evidence from ignored local `jobs/`, when available.

Each run exposes a canonical model display name plus separate `quant`,
`inferenceProfile`, and optional `tag` fields. The explorer provides independent
model, quant, profile, and tag filters; the endpoint model ID remains available
in run details for exact reproducibility.

Verifier excerpts are copied into `data.json`; complete `jobs/` directories do
not need to be published. Public evidence links point to normalized result JSON,
run metadata, task definitions, and copied transcripts in the GitHub repository.
The builder never reads task `solution/` directories.

To publish, configure GitHub Pages to deploy from the `docs/` directory on the
default branch. The generated links assume the repository is
`kyuz0/terminal-bench-mini`; override it for a fork with:

```bash
python3 docs/build_data.py --repository owner/repository
```
