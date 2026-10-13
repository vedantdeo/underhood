# underhood — notes for Claude

Project rules live here rather than in a session's private memory, so they travel with the repo
and reach anyone who clones it. The section on tests is a verbatim mirror of a global preference in
`~/.claude/CLAUDE.md`, which is machine-local and backed up by nothing; this copy is the durable one.

**In this repo**, the tests are the spec and come first: they are written red, ordered by
`DEPENDENCY_ORDER` in `tests/conftest.py`, and the bodies are written against them. A new test file
joins that tuple, or `tests/test_spec_order.py` fails.

## New features ship with their tests

**The rule: a push that adds a feature carries the tests that pin it down, in the same push.** Not a
follow-up commit, not "tested by hand", not covered by a live run alone. Comprehensive means every
behaviour the feature promises has a test that fails when that behaviour breaks:

- The ordinary path, each branch the code takes on its input, every refusal, and every error it
  turns into data rather than raising.
- Each guard — a lock, a ceiling, a validation, a cancel — has a test that fails with the guard
  removed. Check by deleting it and running the suite, not by reading the test.
- A test that cannot fail is not cover. Where a property cannot be shown (a race the GIL hides),
  drop the test and say so in the decision record, rather than keep one that passes either way.
- Paid or slow behaviour is tested against fakes. A live test is extra cover, never the only cover.
- Before proposing a push, name the feature's behaviours and the test that holds each one; a
  behaviour with no test is a gap to close or to state, never to leave silent.

**Why:** an untested feature is a claim. The first refactor breaks it without a sound, and the next
reader cannot tell what it was meant to do. Written alongside the code, the tests are the cheapest
spec there is; written later, they describe whatever the code happens to do by then.

**What is not a feature:** a docs or data change, a config value moved, an experiment run, a
rename. A bug fix is not a feature either, but it brings the test that would have caught it.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

## Fine-tuning lives in underhood; model calls live in aigent

**The rule (2026-10-12):** everything that trains or prepares training for a model lives in
underhood. Every call to a hosted LLM goes through aigent. aigent grades.

- **underhood:** label validation, augmentation, building conversations and the train/valid split
  (`underhood.tooldata`), training, inference on our own models, and the GPU and Kaggle machinery.
- **aigent:** the task (schema, prompt, tool, grader), labelling with Claude through `llm/` and its
  budgets, the test sets, grading predictions and the fault census.
- **The seam:** aigent exports plain JSON (`answers.jsonl`, `task.json`) for underhood to build
  from. underhood writes predictions for aigent to grade. underhood never imports aigent, and has
  no grader of its own.

**Why:** one place per concern. Paid calls stay behind the machinery built to admit them, and a
fine-tune is scored by the same grader as the agent that uses it.

## Jobs that finish live in underhood; servers that stay up live in hostllm

**The rule (2026-10-13):** underhood runs work that ends — training, evals, censuses — and the
compute for it (`underhood.gpu`: Kaggle kernels, the Hub, run pricing). hostllm runs servers that
stay up: the engine, its deployments (Modal, JarvisLabs Serverless) and their benchmarks. underhood
publishes adapters and names what to host; hostllm imports nothing from it.

**Why:** a batch kernel that runs once and exits shares little with an endpoint that starts on
demand, keeps weights warm and is measured under load. If hostllm later needs part of
`underhood.gpu`, lift out that generic part then, shaped by a real second user.

## `main` only takes pull requests

Since 2026-10-09 this repo and aigent are public, and `main` in each is guarded by a ruleset, "main
via PR": no direct push, no force push, no deletion, and `mains / lint · types · tests` passes before
a merge. No approval is required, because GitHub never lets an author approve their own PR: Vedant's
merge is the approval. Nobody bypasses the ruleset. aigent's `CLAUDE.md` holds the full rule.

**How to apply:** "push" means push a branch and open a PR with `gh pr create`, each with its own
confirmation. Never merge a PR — that click is Vedant's.

**While Vedant is the only contributor**, PRs come from one long-lived branch, `vedant`: work on it,
push it, open the PR from it, and after a merge carry on from the same branch. Merges are merge
commits only, which keeps `vedant` inside `main`'s history. When a second person joins, this
paragraph goes and each change gets its own branch.
