# Model Card


Customer Churn Analytics Platform – Part 3: Churn Prediction Model

---

# Model Overview

This machine learning model predicts whether a customer is likely to churn within the next 60 days.

The model is designed to help customer success, marketing, and retention teams identify customers who may leave so that proactive action can be taken.

---

# Intended Use

The model should be used for:

* Customer retention planning
* Marketing campaign targeting
* Customer success prioritization
* Churn-risk monitoring
* Business intelligence reporting

The model is intended to support decision-making, not replace it.

---

# Model Type

Primary Model:

XGBoost Classifier

Baseline Model:

Logistic Regression

The XGBoost model was selected because it delivered stronger predictive performance on customer churn data.

---

# Input Features

The model uses customer behavioral and engagement information such as:

* Recency
* Frequency
* Monetary value
* Support ticket count
* Return rate
* Customer tenure
* Campaign engagement
* Product activity

All features are calculated using information available before the prediction date.

---

# Prediction Output

The model returns:

### Churn Probability

A value between 0 and 1 indicating likelihood of churn.

### Predicted Class

0 = Customer Expected To Stay

1 = Customer At Risk Of Churn

### Risk Category

* Low Risk
* Medium Risk
* High Risk

---

# Model Performance

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC

Key observations:

* Strong separation between churners and non-churners
* Good performance on imbalanced data
* Business-focused threshold optimization

Detailed metrics are available in:

metrics.json

---

# Explainability

Model predictions are explained using SHAP (SHapley Additive Explanations).

SHAP helps identify:

* Why a customer received a high churn score
* Which features influenced predictions
* Global feature importance

This improves transparency and trust in model outputs.

---

# Threshold Selection

The default classification threshold of 0.50 was evaluated and compared with alternative thresholds.

The final threshold was selected based on:

* Retention cost
* Missed churn risk
* Business value

Priority was given to identifying potential churners rather than maximizing raw accuracy.

---

# Limitations

The model has several limitations.

### Historical Data Dependence

Predictions rely on historical behavior and may not reflect future market changes.

### External Events

The model cannot observe:

* Competitor promotions
* Economic changes
* Personal customer circumstances

### Data Quality

Prediction quality depends on the quality of available data.

### Behavioral Changes

Customers may change behavior unexpectedly.

---

# Ethical Considerations

The model should not be used to:

* Deny services
* Restrict customer access
* Make fully automated decisions

Predictions should support customer retention efforts only.

---

# Human Oversight

Human review is recommended for:

* High-value customers
* Borderline predictions
* Strategic retention decisions

The model should assist decision-makers rather than replace them.

---

# Monitoring Requirements

After deployment, the following should be monitored:

* Data drift
* Prediction drift
* Recall performance
* API reliability
* Business outcomes

Retraining should occur if performance declines significantly.

---

# When The Model Should Not Be Used

The model should not be used:

* Outside the customer population on which it was trained
* As the sole basis for major business decisions
* When required input features are missing or unreliable
* Without periodic monitoring and retraining

---

# Summary

This churn prediction model provides an effective way to identify customers at risk of leaving and supports proactive retention efforts.

When used responsibly and monitored regularly, it can help improve customer retention, increase customer lifetime value, and support data-driven business decisions.
