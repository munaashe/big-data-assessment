# Hotel Reservations Analysis

Reproducible Python analysis for the COM00148M Big Data Analytics formative assessment. The workflow examines hotel cancellation patterns, evaluates cancellation classifiers, and quantifies record uniqueness under quasi-identifier generalization.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The supplied dataset is expected at `data/hotel-reservations.csv`.

## Run

From the project root, run each assessment analysis:

```bash
python -m src.analysis
python -m src.evaluation
python -m src.privacy
```

`src.analysis` produces country-month rates with 95% Wilson intervals (cells with fewer than 50 bookings are omitted), cancellation-rate summaries, lead-time bins, single-feature AUC/Cramer's V association measures, and report-ready figures. It ranks numeric and categorical effects on a common 0-to-1 scale using `2 × |AUC - 0.5|` for numeric attributes and Cramer's V for categorical attributes, while preserving raw AUC in the output. These are descriptive associations, not causal effects; the summary explicitly flags the strong concentration of non-refundable bookings in Portugal.

`src.evaluation` uses an explicit set of booking-time attributes selected from the Section 1 analysis. It excludes outcome fields and variables that may be recorded after booking (`required_car_parking_spaces`, `booking_changes`, and `days_in_waiting_list`). Logistic regression and random forest are compared with a majority-class baseline over 80/20, 70/30, and 60/40 group-aware splits, repeated with five fixed seeds. A paired ablation reruns both classifiers without `country` and `deposit_type` on the same splits; `output/ablation_comparison.csv` reports the metric changes explicitly. Identical reduced-feature profiles are kept together to avoid profile overlap in either feature set. Outputs include run-level results, mean and standard deviation, confusion matrices, feature effects, numeric-only signed logistic coefficients, and precision/recall/false-alert/missed-cancellation tradeoffs for thresholds from 0.30 to 0.70. Group-aware splits yield approximate rather than exact percentages; actual row counts are reported.

`src.privacy` measures row-weighted k-anonymity for exact date/country/party-size quasi-identifiers and after generalizing arrival to month, country to continent, and party size to bands. It writes aggregate statistics only. These estimates do not prove re-identification or guarantee privacy: k-anonymity does not address attribute disclosure and is not a substitute for lawful processing, access controls, or a disclosure-risk review.

Generated CSV summaries, JSON summaries, and figures are saved under `output/`.

## Method notes

The preprocessing audit records removed negative-ADR and zero-guest rows, merges `CN` into `CHN`, merges `Undefined` meals into `SC`, and caps ADR at its observed 99.5th percentile for descriptive analysis. In model evaluation, the ADR cap is learned independently within each training fold to avoid using held-out values. Four missing child counts are treated as zero only when checking guest totals; other missing predictors are imputed within each training fold. Repeated rows are retained because there is no reservation ID to prove they are erroneous duplicates. The analysis outputs document all cleaning counts and thresholds.
