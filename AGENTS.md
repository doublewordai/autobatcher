# autobatcher — Agent & Developer Guide

This repo contains two SDKs that implement the same concept: a drop-in OpenAI
client replacement that transparently batches requests via the Batch API.

```
autobatcher/
├── python/          # PyPI: autobatcher
│   ├── src/autobatcher/
│   ├── tests/
│   └── pyproject.toml
├── typescript/      # npm: autobatcher
│   ├── src/
│   └── package.json
└── .github/workflows/
```

## Core invariant

Both SDKs must behave identically from the caller's perspective. A user should
be able to read the Python docs and know exactly how the TypeScript version
works, and vice versa.

Specifically:

- **Same constructor parameters** (snake_case in Python, camelCase in TypeScript)
- **Same defaults** — `mode: "async"`, `batchSize: 1000`, `batchWindowSeconds: 10`,
  `pollIntervalSeconds: 5`
- **Same batch lifecycle** — queue → JSONL → file upload → batch create → poll → fetch results → resolve
- **Same partial-result support** — `X-Incomplete` / `X-Last-Line` headers with `?offset=` query param
- **Same intercepted endpoints** — `chat.completions.create()`, `embeddings.create()`
- **Same subclass contract** — `BatchOpenAI` extends the platform's OpenAI client (`AsyncOpenAI` in Python, `OpenAI` in TypeScript), passes `isinstance`/`instanceof` checks, and leaves all non-batched methods (files, batches, models, etc.) untouched

If you change behaviour in one SDK, check whether the other needs the same change.

## Development

### Python

```bash
cd python
pip install -e ".[test,serve]"
pytest tests/ -v
```

- Python 3.12+
- Type checking: `ty check` (via the `ty` package)
- Linting: configured in `pyproject.toml` (ruff)
- The `serve` extra installs `aiohttp` for the HTTP proxy

### TypeScript

```bash
cd typescript
npm install
npm run typecheck   # tsc --noEmit
npm run build       # tsup → dist/
```

- Node 18+ (but the core `client.ts` is runtime-agnostic — no `node:` imports,
  works in Cloudflare Workers, Deno, Bun)
- `serve.ts` and `cli.ts` use `node:http` and `node:util` — Node/Bun/Deno only
- No runtime dependencies beyond `openai`

## Batch API protocol

Both SDKs talk to the same OpenAI-compatible Batch API:

1. **Upload JSONL** — `POST /v1/files` with `purpose: "batch"`
   ```jsonl
   {"custom_id":"uuid-1","method":"POST","url":"/v1/chat/completions","body":{...}}
   {"custom_id":"uuid-2","method":"POST","url":"/v1/embeddings","body":{...}}
   ```

2. **Create batch** — `POST /v1/batches`
   ```json
   {"input_file_id":"file-abc","endpoint":"/v1/chat/completions","completion_window":"1h"}
   ```

3. **Poll** — `GET /v1/batches/{id}` until `status` is `completed`, `failed`, `expired`, or `cancelled`

4. **Fetch results** — `GET /v1/files/{output_file_id}/content`
   - Doubleword extension: supports `?offset=N` with `X-Incomplete: true` and
     `X-Last-Line: N` response headers for partial result streaming on large batches

## Release process

Both SDKs use [release-please](https://github.com/googleapis/release-please)
with conventional commits. The config is in the repo root:

- `release-please-config.json` — defines both packages
- `.release-please-manifest.json` — tracks current versions

### Tags

| Package | Component | Tag format | Example |
|---------|-----------|------------|---------|
| Python | `autobatcher` | `autobatcher-v{version}` | `autobatcher-v0.6.2` |
| TypeScript | `autobatcher-ts` | `autobatcher-ts-v{version}` | `autobatcher-ts-v0.1.0` |

The `component` field controls the tag prefix. `package-name` is `autobatcher`
for both (this is the published name on PyPI/npm, not the tag).

### Workflow

1. Push conventional commits to `main`
2. Release-please creates per-package PRs bumping versions and changelogs
3. Merge a release PR → creates a GitHub release with the appropriate tag
4. The tag triggers the publish workflow:
   - `autobatcher-v*` → `publish-python` job → PyPI (trusted publishing)
   - `autobatcher-ts-v*` → `publish-typescript` job → npm (via `NPM_TOKEN` secret)

### Conventional commits

Commits that only touch `python/` will only create a Python release PR.
Commits that only touch `typescript/` will only create a TypeScript release PR.
Commits that touch both (or root files) may create both.

```
feat: add responses API support        → minor bump
fix: handle empty JSONL response        → patch bump
feat!: change default completion window → major bump
chore: update CI config                 → no release
```

### Independent versioning

The two packages version independently. Python is at `0.6.x`, TypeScript starts
at `0.1.0`. There is no requirement to keep them in sync — they share the same
behaviour but release on their own cadence.

## CI

The CI workflow (`.github/workflows/ci.yml`) runs on every push to `main` and
on PRs. It has two independent jobs:

- **`python`** — `working-directory: python` → install, type check, test
- **`typescript`** — `working-directory: typescript` → install, typecheck, build

Both must pass before merging.

## Testing

### Python
Tests are in `python/tests/` and use `pytest` with `pytest-asyncio`. They mock
the OpenAI SDK's HTTP layer — no real API calls.

### TypeScript
No tests yet. When adding tests, follow the same pattern: mock `fetch` or the
OpenAI client methods, verify the queue → flush → resolve lifecycle.

## Key files

| File | Purpose |
|------|---------|
| `python/src/autobatcher/client.py` | Python `BatchOpenAI` (~900 lines) |
| `python/src/autobatcher/serve.py` | Python HTTP proxy |
| `python/src/autobatcher/__main__.py` | Python CLI |
| `typescript/src/client.ts` | TypeScript `BatchOpenAI` |
| `typescript/src/serve.ts` | TypeScript HTTP proxy |
| `typescript/src/cli.ts` | TypeScript CLI |
| `release-please-config.json` | Release-please package config |
| `.release-please-manifest.json` | Current version tracking |
