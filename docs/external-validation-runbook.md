# External Validation Runbook

This file is for the part that cannot be faked in code: real user validation.

## Goal

Validate that the tool saves time compared to framework-first alternatives.

## Recruit

- 3 to 10 applied LLM engineers
- each with a small real seed set
- ideally across different verticals

## Ask Each Tester To Do

1. install the tool
2. run one generation workflow
3. inspect the report
4. say whether the output is usable
5. say where the friction was

## Questions

- How long did it take to first useful output?
- Did you need to read code or docs deeply?
- Was the dataset usable without manual cleanup?
- Was the quality report understandable?
- Would you choose this over a framework for this task?

## Success Threshold

- at least 3 testers complete a run
- at least 2 say time-to-value is better than alternatives
- repeated friction points are written down and fixed
