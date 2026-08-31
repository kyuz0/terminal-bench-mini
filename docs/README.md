# Results explorer

`docs/` is a dependency-free static results explorer intended for GitHub Pages.
It reads `data.json` in the browser and does not require an API, database, or
application server.

Regenerate the dataset whenever committed benchmark results change:

```bash
python3 docs/build_data.py
```

Preview it locally from the repository root:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/docs/`. Opening `index.html` directly through
`file://` will not work because browsers block the JSON fetch.

The builder reads:

- normalized run and attempt data from `results/`;
- task descriptions, metadata, and evaluated instructions from `tasks/`;
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
`kyuz0/terminal-bench-local`; override it for a fork with:

```bash
python3 docs/build_data.py --repository owner/repository
```
