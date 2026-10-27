# Model card — synthetic Random Forest demonstration

**Intended use:** engineering demonstration of origination-time 12-month first-90+DPD
probability estimation and human credit-risk decision support. **Not intended:** loan
approval/rejection, customer adverse-action reasons, pricing, real underwriting or fairness
certification without independent validation. No real borrowers were used.

**Development:** deterministic structural synthetic generator (seed 42); median imputation,
missing indicators, one-hot categories, three origination-derived ratios. Bounded temporal
CV compares regularized logistic regression, bootstrap-aggregated Random Forest and histogram
boosting. Sigmoid/isotonic calibration fitted within training folds, chosen on validation.
RF is the required primary family, not assumed best by measurement. Metadata retains
configuration, dataset/schema versions, dependency versions, hashes, timestamp and commit
when available. Exact data, versions and split semantics are in the run manifest.

**Evaluation:** final evidence is in [verification](verification.md) and the generated
comparison report. Chronological evaluation has 12-month maturity embargoes; all immature
outcomes are excluded. Validation selects calibration and cost thresholds; test is used
only after choices freeze. Synthetic generator assumptions fundamentally limit external
validity; strong metrics would not imply real-world predictive skill. IID bootstrap
intervals omit temporal/dependence, generation and specification uncertainty.

**Explanation:** transformed-feature impurity and raw-feature permutation importance,
plus local calibrated-probability reference replacement sensitivity. Correlations and
off-manifold changes undermine causal interpretations. No legally validated reason codes
or SHAP additivity claims. Risk bands and review costs are provisional configuration.

**Fairness:** protected attributes are absent and no representative audit dataset is
provided; fairness has not been established. Group metrics utility requires authorized
separate data and suppresses undersupported groups. Real protected-group performance,
proxy analysis, sampling/selection bias, historical policy effects and missing-not-at-random
mechanisms remain to be evaluated. Approved applicants alone may create selection bias.

**Deployment controls:** candidate models cannot serve by default; externally trusted
manifest digest, compatible runtime, bearer auth, bounded execution, safe errors and
readiness checks are implemented. Synthetic models are blocked from production promotion.
Gateway identity/TLS/rate limits, security assessment, independent model validation,
monitoring operations, governance and jurisdiction-specific review are mandatory before
real use. Consult [operations](operations.md) for lifecycle, rollback and monitoring.
