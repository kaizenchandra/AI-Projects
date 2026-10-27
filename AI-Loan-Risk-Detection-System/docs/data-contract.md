# Prediction and data contract

At loan origination, estimate the first 90+ days-past-due event within the following
12 calendar months. Monetary values must use one model-specific documented currency;
the synthetic example uses generic currency units. Rates are ratios, not percentages.
The pipeline is decision support; it cannot approve or reject applications.

All eleven input keys are mandatory; explicit JSON null requests learned imputation.
Unknown fields are rejected. Numeric JSON strings, booleans, NaN/infinity, fractions in
integer fields, and out-of-range values are rejected by the API. CSV ingestion safely
coerces numeric strings but rejects boolean numeric values, matching the API. Real data
must meet the same semantic units.

| Feature | Allowed range / type | Origination definition |
|---|---|---|
| annual_income | 1–10,000,000 | verified annual income |
| loan_amount | 100–2,000,000 | requested principal |
| loan_term_months | integer 6–360 | requested term |
| employment_years | 0–60 | documented employment tenure |
| debt_to_income | 0–3 | existing monthly obligations / monthly income |
| credit_utilization | 0–2 | utilized / available credit; over-limit allowed |
| credit_history_years | 0–80 | history available at origination |
| monthly_debt_obligations | 0–1,000,000 | existing documented monthly debt |
| prior_delinquencies | integer 0–100 | prior delinquencies known at origination |
| active_credit_accounts | integer 0–100 | known active accounts |
| repayment_history | string 1–40 chars | current / previous_late / limited in generator |

Unseen categories are supported by one-hot encoding with unknowns ignored. Upstream
systems must enforce their approved vocabulary and prohibit arbitrary PII in this field.
Protected traits, location, names, identity numbers, interest/pricing terms, future
payments and outcomes are excluded. Removing protected traits does not establish fairness.

Training CSV adds exactly `loan_id`, `borrower_id`, `origination_date`, `observation_end`,
`default_date`, `default_12m`. Dates parse as UTC, then normalize to timezone-naive UTC.
Blank default_date means no observed 90+ DPD event. Duplicate identical rows are removed;
conflicting loan IDs fail. Target values must equal the event-derived mature target.
The feature allowlist is enforced at every preprocessing and inference boundary.

| Engineered feature | Formula | Missing/extreme handling |
|---|---|---|
| loan_to_income | loan_amount / annual_income | null propagates; cap at 100 |
| income_to_obligation | annual_income / 12 / max(monthly_debt_obligations, 1) | null propagates; cap at 100 |
| monthly_principal_to_income | loan_amount / loan_term_months / (annual_income / 12) | null propagates; cap at 100 |

Monthly principal is a capacity proxy, not a fully amortized installment. Interest is
excluded because its availability depends on the pricing decision. Source ranges prevent
zero income and term denominators. The one-unit obligation floor is a documented
engineering convention, not underwriting policy. Median imputation, missing indicators,
scaling and category encoding are fitted separately inside each training fold.

Splits: train through 2016; validation 2018–2019; test 2021–2023. Entire 2017 and 2020
origination cohorts are embargoed so earlier labels mature before later scoring starts.
Within training, expanding temporal CV applies the same 12-month maturity rule. Borrowers
crossing outer partitions fail; within folds they are purged from earlier fit rows. Dates
and IDs never enter features. No random stratified fallback silently replaces temporal CV.
This is conservative and data-hungry. Configure dates only before inspecting test outcomes.
