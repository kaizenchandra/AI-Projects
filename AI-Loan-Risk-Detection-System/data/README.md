# Data handling

No real loan data is bundled. `loan-risk generate` creates explicitly synthetic data,
with unique artificial loan/borrower IDs and a known, nonlinear sampling process.
Do not use demonstration metrics for real lending claims. CSV output is gitignored.

To use real data, establish licensing and authorized purpose, document currency and
as-of origination semantics, implement an upstream approved field mapping, then meet
[the contract](../docs/data-contract.md). Set `synthetic: false` and a genuine immutable
dataset identifier in a new training configuration. Do not relabel synthetic data as real.
Store identifiable joining data and fairness audit attributes separately with access controls.
IDs here are pseudonymous grouping metadata; they never enter predictive preprocessing.

`default_date` is the first **90+ DPD** event, not the first missed payment. A provider must
attest that ascertainment is complete through `observation_end`; code cannot prove this.
Target-negative loans require twelve calendar months of follow-up. This implementation
conservatively excludes all immature loans, including early positive events, to avoid
asymmetric cohort selection. Censoring, refinancing and early termination need independent
business rules before real-data adoption. Missing outcomes cannot be assumed non-default.
