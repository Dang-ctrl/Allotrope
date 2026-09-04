# Allotrope — instructions for Claude

SIH 2026, PS SIH26061: safe AI energy management for polar research station
microgrids. Software only.

## Read first

1. **[context.md](context.md)** — current state, environment, next steps, open
   questions. Always read this before starting work.
2. **[docs/PROJECT_BIBLE.md](docs/PROJECT_BIBLE.md)** — what the project is, the
   architecture, every design decision and its reasoning.
3. **[docs/calibration.md](docs/calibration.md)** — where every number came from.
   Read before quoting any figure.

## Keep the documentation current

**At the end of any session that changes the state of the project, update
`context.md`** — phase, test count, commits, decisions taken, questions opened or
closed.

**Update `docs/PROJECT_BIBLE.md` whenever the architecture, parameters, results,
roadmap or claims change.** When a result changes, change the number *and*
re-check the bible's section 12, "What this project is not entitled to claim". That
list is the part most likely to go quietly stale and the part that matters most.

Update `README.md` when a headline result or the phase table changes. These are
not optional housekeeping — they are how the next session avoids re-deriving
context that was already paid for.

## Environment

Python **3.11** venv at `.venv`. The machine default `python` is 3.13 and **has
no pip**; 3.11 does. Always invoke `.venv/Scripts/python.exe` explicitly.

```bash
.venv/Scripts/python.exe -m pytest -q
```

Writing Python via a bash heredoc **fails when the content contains backticks**
(common in docstrings). Use the Write tool for Python; heredocs are fine for YAML
and Markdown.

## How this project works

- **Every physical parameter lives in station YAML, never in code**, tagged
  `[public]` / `[derived]` / `[assumed]`.
- **Every claim in the README is reproducible by a script in `scripts/`**, and
  the invariants behind it are asserted in `tests/`.
- **The safety layer's guarantee is the project's central claim.** Test it with
  adversarial policies, never merely sensible ones. Three real bugs were found
  that way and none would have been found by random testing alone.
- **Do not overclaim.** Where the evidence stops, say so — in the README, in
  commit messages, and to the user. The bible’s section 12 is the canonical list.
- **No personal data in the repo.** It is public. The SIH deck's team slide
  carries registration numbers, personal emails and mobile numbers; the `.pptx`
  is gitignored and those details stay out of all documentation.

## Commits

Explain *why*, including bugs found and claims deliberately not made. Do not push
without asking — the repo is public.
