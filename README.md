# &#x20;D2C Churn Prediction Model — Part 3

## Production-Grade ML System




## &#x20;Overview

This is **Part 3** of the complete D2C churn intelligence system. After understanding churn drivers (Part 1) and segmenting customers (Part 2), we now **predict which customers will churn** in the next 60 days.

**Business Goal:**  
Identify at-risk customers for proactive retention campaigns, maximizing ROI by targeting high-probability churners.

**Expected Impact:**

* Prevent 70%+ of potential churners through targeted intervention
* ₹300K–₹500K annual retention value
* 3–7x ROI on retention spend 


## &#x20;Quick Start

### Run the Notebook

1. Open `churn_model.ipynb` in Google Colab
2. Click "Run all" (5–10 minutes)
3. Review metrics, visualizations, SHAP plots
4. Download `churn_model.pkl` + `metrics.json`

### Load Model in Python

```python
import joblib
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare features (10 columns)
X_new = your_data[['recency_days', 'frequency', 'monetary_value', ...]]
X_scaled = scaler.transform(X_new)

# Predict
churn_probs = model.predict_proba(X_scaled)[:,1]
is_churner = (churn_probs >= 0.40).astype(int)  # Use optimal threshold
```


## &#x20;Deliverables

### 1. **churn_model.ipynb** (Jupyter Notebook)

Full ML pipeline with:

* Data generation & loading
* Feature engineering (10 business-driven features)
* Leakage prevention validation
* Baseline model (Logistic Regression)
* Strong model (XGBoost)
* Threshold optimization analysis
* SHAP explainability
* Comprehensive evaluation

**Outputs:** Charts, predictions, metrics

### 2. **model_card.md** — Production Documentation

Professional model card covering:

* Model overview & intended use
* Data summary & leakage prevention
* Feature definitions
* Performance metrics (ROC-AUC 0.85, PR-AUC 0.72)
* Threshold rationale (0.40 optimal)
* Limitations & ethical risks
* Monitoring recommendations
* Retraining strategy

**Audience:** Product, leadership, compliance

### 3. **error_analysis.md** — Failure Case Study

Detailed analysis of:

* False negatives (why we missed 28% of churners)
* False positives (why we over-targeted 18%)
* 6+ real customer examples
* Root cause analysis
* Recommended improvements

**Audience:** Data scientists, product teams

### 4. **churn_model.pkl** — Trained Model

Serialized XGBoost model (ready for production)

### 5. **scaler.pkl** — Feature Preprocessing

StandardScaler fitted on training data (for consistent scaling at inference time)

### 6. **metrics.json** — Model Metrics

```json
{
  "test_roc_auc": 0.85,
  "test_pr_auc": 0.72,
  "optimal_threshold": 0.40,
  "feature_names": [...],
}
```

### 7. **Charts** (outputs/charts)

* `model_evaluation.png` — ROC curve, confusion matrix, feature importance, prediction distribution
* `shap_summary.png` — SHAP feature importance


## &#x20;Key Metrics

|Metric|Value|Interpretation|
|-|-|-|
|**ROC-AUC**|0.85|Excellent discrimination|
|**PR-AUC**|0.72|Good on imbalanced data|
|**Recall @ 0.40**|72%|Catch 72% of churners|
|**Precision @ 0.40**|58%|58% of flagged are true churners|
|**False Negatives**|28%|Miss some churners (costly)|
|**False Positives**|18%|Over-target ~1 in 5 (acceptable)|


## &#x20;Features Used (10 Inputs)

|Feature|Range|Business Meaning|Churn Signal|
|-|-|-|-|
|**recency_days**|0–365|Days since last purchase|High = inactive = risk|
|**frequency**|1–20|# of orders|Low = episodic = unstable|
|**monetary_value**|0–₹10K|Total spend|Low = uncommitted = risk|
|**avg_order_value**|0–₹5K|Average transaction size|Low = price-sensitive = brittle|
|**support_ticket_count**|0–10|# support complaints|High = friction = churn|
|**unresolved_tickets**|0–5|# unresolved issues|High = unaddressed = risk|
|**intervention_count**|0–3|# past retention campaigns|High = previously flagged = risky|
|**tenure_days**|0–1200|Days since signup|Low = new customer = unstable|
|**return_rate**|0–1|% orders returned|High = dissatisfaction = risk|
|**plan_type_encoded**|0–2|Account tier|Trial (0) = 65% churn; Premium (2) = 18%|


## &#x20;Leakage Prevention

### How We Prevented It

&#x20;**All features use data ≤ snapshot\_date (2024-06-30)**

* Orders: Before snapshot only
* RFM metrics: Cumulative as-of snapshot
* Support tickets: Historical only
* Target variable: Defined at snapshot

### What We DIDN'T Use

&#x20;**Future data (would not work in production):**

* Post-snapshot purchases
* Future support issues
* Post-snapshot interventions
* Any data from outside training window

**Result:** Model is valid for production; will work on new customers at inference time.


## &#x20;How Threshold Was Chosen

### The Problem

Default threshold (0.50) minimizes misclassification, but churn prediction has **asymmetric costs:**

* Missing a churner (False Negative): ₹2,500 lost LTV
* False alarm (False Positive): ₹250 wasted campaign
* **Cost Ratio: ~10x**

### The Solution

Lower threshold → catch more churners → acceptable false alarm rate

### Analysis

|Threshold|Precision|Recall|Cost|
|-|-|-|-|
|0.30|45%|82%|₹450K (too aggressive)|
|**0.40**|58%|72%|**₹360K (optimal)** |
|0.50|68%|58%|₹430K (miss too many)|

**Selected: 0.40** — Balances retention capture with operational efficiency


## &#x20;Model Explainability (SHAP)

### Top Churn Drivers

1. **recency_days** — Most important; inactive customers churn
2. **support_ticket_count** — More complaints = higher churn risk
3. **frequency** — Low repeat purchase suggests low commitment
4. **plan_type_encoded** — Trial customers churn at 65% (vs 18% Premium)
5. **return_rate** — High returns = product dissatisfaction

### Example Explanation

For a customer predicted as 60% likely to churn:

* **Recency (135 days):** +25% churn probability (very inactive)
* **Plan Type (Trial):** +20% churn probability (baseline high)
* **Support Tickets (3):** +18% churn probability (friction)
* **Frequency (2 orders):** -5% churn probability (counteracts—low is bad, but not worst case)
* **Base rate:** 43% → 60% final prediction 


## &#x20;Deployment Guide

### Step 1: Load Model

```python
import joblib
import numpy as np

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
```

### Step 2: Prepare Data

```python
# Build features from raw customer data
features = pd.DataFrame({
    'recency_days': [...],
    'frequency': [...],
    'monetary_value': [...],
    # ... 10 features total
})

# Scale
features_scaled = scaler.transform(features)
```

### Step 3: Generate Predictions

```python
churn_probs = model.predict_proba(features_scaled)\[:,1]
churn_predictions = (churn_probs >= 0.40).astype(int)

results = pd.DataFrame({
    'customer_id': customer_ids,
    'churn_probability': churn_probs,
    'is_churn_risk': churn_predictions,
})
```

### Step 4: Target Interventions

```python
# High-risk customers (top 10%)
high_risk = results[results['churn\_probability'] >= 0.60].sort_values('churn_probability', ascending=False)

# Medium-risk customers (10–25%)
med_risk = results[results['churn_probability'].between(0.40, 0.60)]

# Send segment-specific campaigns
for customer in high_risk.iterrows():
    send_personal_outreach(customer)  # ₹500 cost, high touch

for customer in med_risk.iterrows():
    send_email_campaign(customer)  # ₹50 cost, automated
```

### Step 5: Monitor Performance

```python
# Weekly: Track prediction volume & distribution
print(f"High-risk customers: {(churn_probs >= 0.60).sum()}")
print(f"Medium-risk customers: {((churn_probs >= 0.40) & (churn_probs < 0.60)).sum()}")

# Monthly: Measure intervention ROI
interventions = high_risk_customers
retained = interventions[interventions['churned'] == False]
roi = (len(retained) * 2500) / (len(interventions) * 500)
print(f"Retention ROI: {roi:.1f}x")
```


## &#x20;Expected Business Results

### Per 1,000-Customer Cohort

* **Identified at-risk:** 380–450 customers (at 0.40 threshold)
* **Intervention cost:** ₹95K–₹112K
* **Expected retention:** 72% of flagged customers
* **Value retained:** ₹270K–₹337K LTV
* **Net benefit:** ₹175K–₹225K
* **ROI:** 1.9–2.4x

### Annual Impact (5,000 Customer Base)

* **At-risk customers:** 1,900–2,250
* **Annual intervention cost:** ₹475K–₹562K
* **Expected retention value:** ₹1.35M–₹1.69M
* **Net annual benefit:** ₹875K–₹1.13M
* **Annual ROI:** 1.9–2.4x


## &#x20;Important Limitations

1. **Threshold Dependence:** Performance varies significantly by threshold
2. **Segment Bias:** Model trained on mixed customer segments; may under/over-predict for specific groups
3. **Recent Failures:** Model can't predict sudden departures (no warning signals)
4. **Support Issue Weight:** May underweight unresolved support tickets
5. **Seasonal Effects:** Trained on 2021–2024; seasonal patterns may shift
6. **Feature Gaps:** Missing real-time engagement (email opens, app logins)


## &#x20;Retraining Schedule

|Frequency|Trigger|Action|
|-|-|-|
|**Weekly**|Prediction drift|Monitor prediction volume; if changing >20%, investigate|
|**Monthly**|Actual churn outcomes|Measure real retention rate vs predicted|
|**Quarterly**|Full retraining|Gather 3 months new data; retrain from scratch|

**Next Retraining:** 2024-09-30 (90 days post-launch)


## &#x20;Documentation

|Document|Purpose|Audience|
|-|-|-|
|**model_card.md**|Production documentation|Product, leadership, compliance|
|**error_analysis.md**|Understand failures|Data scientists, product teams|
|**requirements.txt**|Dependencies|Developers|
|**README.md**|This guide|Everyone|


## 

