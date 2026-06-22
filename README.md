# End-to-End Credit Risk Prediction Pipeline

An end-to-end machine learning pipeline built to predict credit default risk using a Decision Tree Classifier. This project processes 45,000 profiles, handles critical real-world data anomalies (such as impossible age and experience outliers), corrects highly skewed distributions, and evaluates performance using business-centric metrics (Precision, Recall, and Confusion Matrix analysis).

## 📌 Project Overview & Business Problem

In the banking sector, loan defaults present a multi-million dollar challenge. When a financial institution issues a loan, they face two key risks:

- **Type I Error (False Positive):** Flagging a creditworthy customer as a default risk, resulting in lost interest revenue.
- **Type II Error (False Negative):** Approving a high-risk applicant who subsequently defaults, resulting in direct capital loss. This is the most expensive mistake in credit risk.

### Objective

To build a highly interpretable binary classification model to predict whether a loan applicant will default (Loan Status of `1`) or pay off their debt safely (Loan Status of `0`).

---

## 🛠️ Complete Pipeline Architecture

```mermaid
graph TD
    A[Raw Data: 45,000 records] --> B[Outlier Handling: Winsorization/Capping]
    B --> C[Feature Engineering: Log Transformation]
    C --> D[Categorical Encoding: Binary Mapping & One-Hot Encoding]
    D --> E[Data Splitting: 80/20 Stratified Split]
    E --> F[Model Training: Decision Tree Classifier]
    F --> G[Model Evaluation: Confusion Matrix & Feature Importances]
