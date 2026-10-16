# Inference-Guided Customer Churn Prediction Pipeline

An enterprise-ready machine learning application that uses rigorous statistical logic to predict customer churn. Built with the Kaggle Telco Customer Churn dataset, this system automatically cleans raw operational logs, tests data shapes, removes redundancies, and trains high-accuracy predictive models inside an isolated, containerized environment.

---

## 📌 Project Overview & Non-Technical Summary

In many modern business settings, software engineers and data analysts build prediction models by throwing raw data directly into artificial intelligence algorithms. While this approach might work initially, it creates a massive blind spot: it assumes the data is perfectly clean, unbiased, and ready for math. In the real world, raw customer data is messy, highly skewed, and filled with overlapping patterns that can cause algorithms to hallucinate or give misleading predictions.

This project solves that exact problem by introducing an **Automated Statistical Gatekeeper** ahead of the machine learning step. Instead of blindly trusting the incoming dataset, the system acts like a scientific assembly line that verifies the structural health of the data at every stage.

### The Scientific Assembly Line:

- **The Ingestion & Common-Sense Check:** The pipeline loads the customer logs and handles missing information using real business logic. For example, if a customer has a service history of `0` months, their total bills paid should logically be `0.0`. Instead of throwing these rows away (which would delete all new sign-ups), the system fixes them automatically.
- **The Shape & Normality Test:** Mathematical algorithms work best when data follows a natural, bell-shaped curve. This module tests the shape of features like monthly bills. If a feature looks heavily lopsided, the system automatically applies a mathematical formula to straighten the distribution out into a clean, balanced shape.
- **The Variety & Group Test:** The pipeline checks if the fluctuations in billing patterns look steady across both retained and churned customer groups. It dynamically selects the correct mathematical test to verify that the group comparisons are completely fair.
- **The Redundancy Filter:** Sometimes features overlap so much that they tell the exact same story. For instance, "Total Charges Collected" is highly redundant if you already know a customer's "Monthly Bill Amount" and how many months they have stayed with the company. The pipeline calculates a relationship score for all features and automatically discards the overlapping ones to keep the model lightweight and stable.
- **The Final Prediction Engine:** Only the cleanest, most reliable, and completely independent customer features are allowed into the final machine learning classifier. The system trains a robust model that carefully handles imbalances (since remaining customers usually outnumber leaving customers) and outputs trustworthy predictions.

---

## 🏗️ Architecture & Module Blueprint

The codebase uses a highly decoupled, modular structure designed for enterprise stability:

```text
📦 production_ml_pipeline/
├── 📂 config/
│    └── config.yaml          # Single source of truth for all rules, metrics, and limits
├── 📂 src/
│    ├── __init__.py          # Exposes directory packages natively
│    ├── ingestion.py         # Raw file parsing, data type cleaning, and business logic fixes
│    ├── normality.py         # Shapiro-Wilk testing and distribution shape smoothing
│    ├── variance.py          # Levene / Bartlett variance consistency checks
│    ├── testing.py           # Automated feature engine selector (Mann-Whitney U routing)
│    ├── multicollinearity.py # Iterative redundancy filtering via Variance Inflation Factor (VIF)
│    └── model.py             # Stratified cross-validation, performance scoring, and asset serialization
├── 📄 main.py                # Master application orchestrator script
├── 📄 requirements.txt       # Fixed library dependencies snapshot
└── 📄 Dockerfile             # Container recipe for deployment isolation
```

---

## ⚙️ Project File Registry

- **`config/config.yaml`**: Houses all decision boundaries, column maps, and target thresholds. No variables are hardcoded into the execution engines; changing a value here changes the behavior of the entire system.
- **`src/ingestion.py`**: Performs type coercion and applies business-driven missing data imputation. It automatically standardizes the target churn labels into clean computer bits.
- **`src/normality.py`**: Runs Shapiro-Wilk data shape checks. If a column is lopsided, it applies a Yeo-Johnson smoothing transformer before standardizing the scale.
- **`src/variance.py`**: Evaluates whether fluctuations remain steady across customer cohorts, dynamically routing data to Levene or Bartlett tests.
- **`src/testing.py`**: Validates whether individual features are mathematically significant predictors, sorting columns using strict non-parametric routing filters.
- **`src/multicollinearity.py`**: Measures multi-variable correlation loops. It actively flags and discards columns (like `TotalCharges`) that register a VIF score greater than `5.0`.
- **`src/model.py`**: Splits records into Stratified 5-Fold cross-validation groups to ensure stable performance metrics. It then fits a final Balanced Random Forest classifier and serializes the assets into reusable binary `.pkl` files.

---

## 🚀 Execution & Quick Start Guide

### Local Environment Verification

To verify the entire analytical assembly line directly on your local system, run the master orchestrator file from your terminal:

```bash
python main.py
```

### Production Deployment via Docker

To isolate the application environment completely from your host machine and ensure it runs identically anywhere, use the following container steps:

1.  **Build the Container Image Layers:**

    ```bash
    docker build -t statistical-ml-pipeline:latest .
    ```

2.  **Execute the Application Using Shared Folders:**
    Since raw operational datasets change daily, we leave the moving data file outside on the host system and mount it to the container dynamically using a folder volume (`-v` parameter):
    ```bash
    docker run --rm -v "%cd%:/app" statistical-ml-pipeline:latest
    ```
    _(Note: If executing this step from a Windows PowerShell terminal instead of a classic Command Prompt, use `-v "${PWD}:/app"` instead)._

---

## 📈 Operational Performance Matrix

When the pipeline finishes running inside Docker, it outputs a cross-validated performance report and isolates the core customer drivers:

- **Mean Accuracy Rate:** `0.7843`
- **Weighted F1-Score:** `0.7740`
- **Mean ROC-AUC Score:** `0.8150`

### Top 5 Customer Churn Drivers Discovered:

- **tenure** (Account Age in Months) — `22.92%` System Importance
- **MonthlyCharges** (Current Billing Rate) — `20.20%` System Importance
- **Contract_Two year** (Long-Term Commitment Hold) — `6.49%` System Importance
- **InternetService_Fiber optic** (High-Speed Fiber Infrastructure) — `4.81%` System Importance
- **PaymentMethod_Electronic check** (Manual Monthly Billing Choice) — `3.95%` System Importance
