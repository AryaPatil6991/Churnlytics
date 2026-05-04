# Churnlytics
### Customer Churn Prediction for a Streaming Service

A machine learning project that predicts customer churn for a streaming service using behavioral, demographic, and billing data — enabling data-driven retention strategies for subscription-based businesses.

---

## Problem Statement

Customer churn is a critical challenge for subscription businesses. This project identifies customers at risk of churning and uncovers the key drivers behind attrition, enabling targeted intervention before customers leave.

---

## Methodology

**Data Engineering**
- Merged and cleaned three datasets covering customer behavior, demographics, and subscription history
- Processed 20,000+ records across 14 features with datetime parsing, type casting, and categorical encoding

**Exploratory Data Analysis**
- Analyzed churn patterns across subscription plans, age groups, billing cycles, primary devices, and geographic regions
- Identified month-to-month contracts and Basic/Standard subscription plans as the strongest churn indicators

**Predictive Modelling**
- Trained a Random Forest Classifier with hyperparameter tuning
- Applied SMOTE to address class imbalance (observed churn rate: ~18%)
- Prioritized recall to ensure maximum detection of at-risk customers

**Retention Strategy**
- Designed customer segmentation based on churn risk and Customer Lifetime Value (CLV)
- Proposed CRM-integrated workflows to flag and engage high-risk customers
- Defined KPIs targeting a 5–10% churn reduction within 6 months

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 92.0% |
| ROC-AUC | 0.91 |
| Recall (Churn Class) | 85.7% |
| Precision (Churn Class) | 66.7% |
| F1-Score (Churn Class) | 0.75 |

- Month-to-month contract customers exhibit the highest churn rates
- Basic and Standard plan subscribers churn significantly more than Premium subscribers
- A recall of 85.7% ensures the majority of at-risk customers are identified early
- The ROC-AUC of 0.91 confirms strong model discrimination between churners and non-churners

---

## Tech Stack

Python, Scikit-learn, Imbalanced-learn, Pandas, Matplotlib, Seaborn, Google Colab

---

## Repository Structure

```
Churnlytics/
│
├── churnlytics.ipynb                    # End-to-end analysis and modelling notebook
├── customer_churn_data.csv              # Customer activity and churn labels
├── demographic_data.csv                 # Customer demographic information
└── subscription_payment_history.csv     # Billing and subscription history
```
