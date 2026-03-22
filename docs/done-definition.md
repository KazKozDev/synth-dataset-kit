# Done Definition

Use this file to avoid calling infrastructure “done” before the product claim is actually closed.

## Strong Product Checklist

### 1. Universal Enough

Done when:

- `3` verticals have real public datasets
- at least `1` of them is not support-adjacent
- the README shows more than one core use case

Not done when:

- the code supports many verticals, but the public proof still centers on one

### 2. Proof Of Results

Done when:

- `2` public uplift stories exist
- each shows before/after metrics
- each links to a generated dataset and training recipe

Not done when:

- `proof`, `finetune`, and `uplift` commands exist, but no public result pages do

### 3. Benchmark Claim

Done when:

- the benchmark proof contains real measured runs
- the conditions are explicit
- the claim uses exact timings and model/provider context

Not done when:

- the workflow is possible, but the timing claim is only descriptive

### 4. Stable Model Surface

Done when:

- the documented model matrix works without known blockers
- local-first and hosted paths both pass the happy path
- repair/retry logic covers the common JSON failure modes

Not done when:

- one or two models work, but the public promise implies broader stability

### 5. Polished UX

Done when:

- command names and outputs feel consistent
- artifact names are short and readable
- failure messages tell the user how to recover
- the happy path has no confusing branch points

Not done when:

- the tool is usable only if the user already thinks like the maintainer

### 6. Strong HF Packaging

Done when:

- dataset pages follow one repeatable structure
- metrics and proof links are visible without scrolling through raw prose
- metadata, tags, and summaries are aligned across pages

Not done when:

- the bundle uploads successfully, but the page does not communicate value

### 7. Real Distribution

Done when:

- the package exists on PyPI
- the product has public dataset pages
- there is at least one public launch artifact outside the repo

Not done when:

- all assets exist, but nobody can reasonably discover them

### 8. External Validation

Done when:

- `3-5` real users tried the product
- their feedback was logged
- at least `3` changes were shipped because of that feedback

Not done when:

- only the builder has run the workflow
