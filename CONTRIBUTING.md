# Contributing to CO₂RE

## Workflow

1. Start from an up-to-date `main`:
   ```bash
   git checkout main
   git pull
   git checkout -b <type>/<short-name>     # e.g. feat/seasonal-analysis
   ```
2. Commit your work in small, focused commits.
3. Before opening a PR, update your branch: `git pull origin main`.
4. Push and open a pull request. Link the issue with `Closes #<number>`.
5. At least one teammate reviews and approves. Do not merge your own PR without review.
6. After merging: `git checkout main`, `git pull`, and delete the branch.

Never push directly to `main`.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary in imperative mood>
```

| Type | Use for |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fix |
| `data` | Adding or updating datasets |
| `docs` | Documentation only |
| `test` | Adding or changing tests |
| `refactor` | Code change without changing behaviour |
| `chore` | Setup, dependencies, housekeeping |

The scope is the component folder: `data-pipeline`, `measurement`, `scheduler`, `panel`, ...
Write commit messages, code, comments, file names and column names in English.

Examples:
- `feat(data-pipeline): add unlicensed generation to carbon intensity`
- `data(data-pipeline): update hourly dataset to 2027-02`
- `docs(measurement): document idle baseline procedure`

## Folder structure

- Code goes in `src/<component>/`, each with its own `README.md`.
- Outputs used by other components go in `data/processed/` (e.g. `hourly.parquet`), with columns documented in the producing component's README.
- Exploration notebooks go in `notebooks/`.
- Measurement experiments live in `test/<experiment>/` (e.g. `test/shelly-plug-test/`) with their scripts, raw logs and results together, until their code moves to `src/measurement/`.

## Data and secrets

- Never commit credentials. `.env` and `.eptr2-tgt` are gitignored; keep it that way.
- Before every commit, check `git status` for files that should not be there.
- The repository is public. Never commit personal data (phone numbers, personal emails), IP addresses, device or machine names, the project proposal PDF, model files (`*.gguf`) or downloaded archives (`*.zip`). Pass local settings such as the plug IP as command-line arguments.
- Raw data that can be downloaded again (e.g. EPİAŞ) is not committed. Measurements that cannot be repeated are committed inside their component folder.
- Dataset versions are marked with git tags (`data-YYYY-MM`). Experiments state which tag they used.
- Raw measurement files are never edited after they are committed. A refactor must not change results: recompute derived tables from the raw data and confirm they are identical before committing.
- Hardware measurements (GPU clock or power changes, plug logging) are started by a person, never by an agent.

## AI coding agents

Agents (Claude Code, Codex, Cursor, ...) follow the same rules. Do not let an agent read `.env` or other credential files, and review every change an agent makes before committing. Claude Code permissions are set in `.claude/settings.json`.
