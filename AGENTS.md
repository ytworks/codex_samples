# Repository Guidelines

> This repository is intentionally minimal. Use this guide to add samples or projects consistently and keep contributions easy to review.

## Project Structure & Module Organization

- Root: `README.md`, `LICENSE`, `.gitignore`, and project docs.
- Single project: place code in `src/` and tests in `tests/` (mirroring package/module layout).
- Multi-project/samples: use `packages/<name>/` or `examples/<name>/` with each subproject owning its own `src/` and `tests/`.
- Tooling: put helper scripts in `scripts/`; CI lives under `.github/workflows/` when added; small fixtures in `assets/`.

## Build, Test, and Development Commands

Prefer a `Makefile` (or per-subproject scripts) to standardize tasks:

```sh
make setup   # install dependencies for the active project
make build   # compile/build artifacts if applicable
make test    # run all tests with coverage
make lint    # run formatters/linters
```

Language-specific examples (run inside the relevant subproject):

- Python: `pip install -r requirements.txt && pytest -q`
- JS/TS: `npm ci && npm test` (or `pnpm i --frozen-lockfile && pnpm test`)
- Go: `go test ./...`

## Coding Style & Naming Conventions

- Use the language’s standard formatter: Prettier (JS/TS), Black (Python), `gofmt` (Go), `rustfmt` (Rust).
- Keep linters clean; fix or justify disables in-code minimally.
- Naming: Python files `snake_case.py`; JS/TS files `kebab-case.ts`; types/classes `PascalCase`; constants `UPPER_SNAKE_CASE`.

## Testing Guidelines

- Frameworks: `pytest` (Python), Jest/Vitest (JS/TS), `go test` (Go), `cargo test` (Rust).
- Coverage: target ≥80% on changed code; include edge cases and error paths.
- Naming: `tests/test_<unit>_does_<behavior>.py`, `<unit>.test.ts`, or `<pkg>_test.go`.
- Run locally before pushing; ensure tests pass deterministically.

## Commit & Pull Request Guidelines

- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`.
  - Example: `feat(api): add user listing`
  - Example: `fix(ci): pin Node 20 in workflow`
- PRs: clear description, linked issues, steps to test, and screenshots/logs when useful. Keep changes scoped; update docs.

## Security & Configuration Tips

- Never commit secrets; add `.env` to `.gitignore` and provide `.env.example`.
- Pin dependencies (`requirements.txt`, `package-lock.json`, `go.mod`) and avoid unnecessary network calls in tests.
- Prefer reproducible scripts over ad-hoc commands; document new configs in the project README.
