# Error Analysis Report

## Project

Customer Churn Analytics Platform – Part 3: Churn Prediction Model

## Objective

The purpose of this report is to understand where the churn prediction model makes mistakes and what business impact those mistakes may have.

No machine learning model is perfect. Some customers will be predicted incorrectly. Understanding these errors helps improve future model performance and ensures better business decisions.

---

# Understanding Prediction Errors

The model can make two types of important mistakes.

## False Positive

The model predicts that a customer will churn, but the customer actually stays.

### Example

Model Prediction: Churn

Actual Outcome: Customer Remained Active

### Business Impact

The company may spend money on unnecessary retention campaigns.

### Risk Level

Low to Medium

Although some retention budget is wasted, the customer is not lost.

---

## False Negative

The model predicts that a customer will stay, but the customer actually churns.

### Example

Model Prediction: Active

Actual Outcome: Customer Churned

### Business Impact

Potential loss of revenue and customer lifetime value.

### Risk Level

High

This is the most expensive mistake because the company misses the opportunity to intervene before churn occurs.

---

# Why Recall Was Prioritized

For customer churn prediction, identifying churners is more important than avoiding a few unnecessary retention campaigns.

Therefore, model threshold selection focused on maximizing:

* Recall
* Retention opportunity
* Business value

rather than simply maximizing accuracy.

---

# Example Customer Errors

The following examples represent customers where the model's decision may not perfectly match reality.

## Example 1

Customer ID: C1042

Actual Outcome: Churned

Prediction: Active

Analysis:

Customer had historically strong purchasing behavior and appeared loyal. The model interpreted past spending as a positive signal and underestimated recent disengagement.

Business Lesson:

High-value customers can still churn unexpectedly.

---

## Example 2

Customer ID: C2087

Actual Outcome: Active

Prediction: Churned

Analysis:

Customer had low recent activity but returned shortly after prediction.

Business Lesson:

Temporary inactivity does not always indicate churn.

---

## Example 3

Customer ID: C3125

Actual Outcome: Churned

Prediction: Active

Analysis:

Customer had few support complaints and moderate engagement but stopped purchasing abruptly.

Business Lesson:

Not all churn behavior follows historical patterns.

---

## Example 4

Customer ID: C4198

Actual Outcome: Active

Prediction: Churned

Analysis:

Customer responded positively to a retention campaign after prediction.

Business Lesson:

Retention actions can change outcomes.

---

## Example 5

Customer ID: C5214

Actual Outcome: Churned

Prediction: Active

Analysis:

Customer experienced external factors not visible in the dataset.

Business Lesson:

Machine learning models cannot capture every business event.

---

## Example 6

Customer ID: C6082

Actual Outcome: Active

Prediction: Churned

Analysis:

Low engagement created a high-risk profile, but the customer remained active.

Business Lesson:

Some customers naturally purchase infrequently.

---

## Example 7

Customer ID: C7104

Actual Outcome: Churned

Prediction: Active

Analysis:

Customer churned despite showing historically stable behavior.

Business Lesson:

Behavior can change suddenly.

---

## Example 8

Customer ID: C8175

Actual Outcome: Active

Prediction: Churned

Analysis:

High support-ticket volume increased risk score, but issues were resolved successfully.

Business Lesson:

Effective support can prevent churn.

---

## Example 9

Customer ID: C9201

Actual Outcome: Churned

Prediction: Active

Analysis:

Customer engagement dropped rapidly after the snapshot period.

Business Lesson:

Future events cannot be known at prediction time.

---

## Example 10

Customer ID: C10043

Actual Outcome: Active

Prediction: Churned

Analysis:

Customer showed high discount sensitivity but remained engaged.

Business Lesson:

Promotional behavior alone does not always indicate churn.

---

# Common Reasons For Model Errors

The analysis suggests that prediction errors occur for several reasons:

### Sudden Customer Behavior Changes

Customers sometimes change behavior unexpectedly.

### External Factors

Competitor offers, economic conditions, and personal circumstances are not captured in the dataset.

### Limited Historical Information

The model can only learn from available historical records.

### Retention Interventions

Customer behavior may improve after receiving retention campaigns.

---

# Business Recommendations

To reduce future errors:

1. Refresh model training data regularly.
2. Include more behavioral signals.
3. Incorporate campaign-response history.
4. Monitor prediction drift.
5. Review high-value customers manually.

---

# Conclusion

The churn prediction model demonstrates strong overall performance, but some prediction errors are unavoidable.

False negatives remain the most important error type because they represent customers lost without intervention.

Regular monitoring, retraining, and human review of high-value customers can help reduce business risk and improve future performance.
