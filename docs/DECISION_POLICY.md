# Decision policy and operational guardrails

This is an educational offline fraud-triage prototype. The model outputs a score in [0, 1], but the model is trained with a class-weighted objective and the score is **not a calibrated probability**. Do not label it as a 94% real-world likelihood of fraud.

- Default demonstration threshold: 0.50.
- Validation-only cost scenario: false negative = 20 cost units; false positive = 1 cost unit. These are illustrative relative weights, not actual money, fees, or issuer loss estimates.
- The threshold minimizing validation cost did not improve the held-out test cost over 0.50 in this run. That is a useful warning about threshold instability with few positive examples and time drift.
- Suggested production path: collect more recent and representative labelled data; calibrate scores on a temporally held-out calibration window; estimate actual fraud loss and review costs with business stakeholders; validate thresholds on a later untouched time window; add manual review and appeal paths; monitor drift and alert volumes; then run shadow mode before any authorization controls.

No transaction should be automatically blocked on this benchmark model alone.
