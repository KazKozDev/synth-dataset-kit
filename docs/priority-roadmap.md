# Priority Roadmap

This is the shortest path from **working MVP** to **strong public product**.

## Order Of Work

1. Prove the product on `3` verticals.
2. Publish `2` end-to-end uplift stories.
3. Lock in the benchmark-proof claim.
4. Remove the last obvious UX friction.
5. Standardize Hugging Face packaging.
6. Move from repo to distribution.
7. Validate with real users.

## Phase 1: Vertical Proof

Goal: stop looking like a `customer-support-only` tool.

Required verticals:

1. Customer support
2. FAQ assistant
3. IT helpdesk

Definition of done:

- each vertical has seeds
- each vertical has holdout/reference data
- each vertical has one generated dataset
- each vertical has one HTML quality report
- each vertical has one eval summary
- each vertical has one Hugging Face dataset page
- README links to all three

Why first:

Without this, every later claim still reads as a narrow demo.

## Phase 2: Uplift Proof

Goal: show that the generated dataset improves something real.

Minimum public set:

1. one customer-support uplift story
2. one non-support uplift story

Each story must include:

- base model
- seed set
- generated dataset
- fine-tune recipe
- holdout/eval data
- before/after metrics
- one short interpretation of the delta

Definition of done:

- two public result pages exist
- each links to the dataset and proof bundle
- each shows a measurable before/after comparison

Why second:

Generated data alone is not a product claim. Improvement is.

## Phase 3: Benchmark Proof

Goal: turn `install in 2 minutes, usable dataset in 10` into a defensible claim.

Required runs:

1. one hosted run
2. one local-first run

Each run must record:

- machine and environment
- install time
- init time
- generation time
- audit/report time
- total time to usable dataset
- output artifact paths

Definition of done:

- the benchmark document contains real timings
- the README links to the results
- the claim is phrased with exact conditions, not vague marketing

## Phase 4: UX Polish

Goal: make the happy path feel like a tool, not an engineering repo.

Must-fix friction:

- naming and output paths stay short and predictable
- health messages are clear
- provider/model errors explain the fix
- defaults stay sane for `sdk go`
- publish flow shows next steps cleanly

Definition of done:

- a new user can finish the happy path without asking what to do next
- no README step requires internal repo knowledge

## Phase 5: Hugging Face Standard

Goal: make every dataset page sell the product by itself.

Every published dataset page should include:

- clear title
- narrow tags
- one-line use case summary
- public quality gate result
- contamination summary
- link to proof bundle
- reproducibility section

Definition of done:

- all public dataset pages follow the same structure
- no page looks like a raw dump of files

## Phase 6: Distribution

Goal: become discoverable outside GitHub.

Minimum launch set:

1. PyPI package
2. three Hugging Face dataset pages
3. one short terminal demo
4. one launch post
5. one README hero section with a single happy path

Definition of done:

- a new person can discover the product without searching the codebase

## Phase 7: External Validation

Goal: confirm that this saves real users time.

Minimum validation set:

- `3-5` external users
- each uses their own seed data
- each reports setup pain, generation quality, and time-to-value

Definition of done:

- user feedback is captured in one summary doc
- at least `3` repo or UX changes can be traced directly to feedback

## What Not To Do Before Phase 1 And 2

- do not add more domains just for breadth
- do not add more export formats unless a showcase needs them
- do not overbuild framework abstractions
- do not spend time on growth messaging before proof exists

## Exit Condition

The product is strong when a public reader can:

1. discover one of the dataset pages
2. understand the value in under `30` seconds
3. trace the dataset back to a reproducible generator workflow
4. see at least one real uplift story
5. believe the product is broader than one demo
