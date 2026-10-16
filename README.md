# Telco Customer Retention GenAI Co-Pilot Pipeline

A production-grade, containerized machine learning application that uses rigorous statistical logic to predict customer churn. Built with the Kaggle Telco Customer Churn dataset, this system automatically handles lopsided metrics, screens out redundant billing tracking, and serves real-time predictive data to an interactive web dashboard and a conversational GenAI Co-Pilot.

---

## 📌 Project Overview & Non-Technical Summary

In standard business setups, analytics teams often build prediction models by throwing raw data directly into machine learning algorithms. While this might look easy, it creates a massive blind spot: it assumes the data is perfectly clean and balanced. In reality, raw customer records are messy, highly skewed, and filled with overlapping patterns that can distort a model’s accuracy or cause artificial intelligence engines to hallucinate.

This project solves that exact problem by introducing an **Automated Statistical Gatekeeper** ahead of the machine learning step. Instead of blindly trusting the incoming data, the system acts like a scientific assembly line that verifies and shapes the data at every stage before any predictions are made.

### The Scientific Assembly Line:

1. **The Common-Sense Check (Ingestion):** The system loads customer logs and handles missing information using real business logic. For example, if a customer has a service history of `0` months, their total bills paid should logically be `$0.0`. Instead of throwing these rows away (which would delete all brand-new sign-ups and introduce bias), the system fixes them automatically.
2. **The Shape & Smoothing Test (Normality):** Mathematical models work best when data follows a natural, bell-shaped curve. The system runs shape checks on variables like monthly bills. If a feature looks heavily lopsided, it applies a mathematical formula to smooth out the distribution into a clean, balanced shape.
3. **The Variety Check (Variance Consistency):** The pipeline evaluates whether the fluctuations in billing patterns look steady across both retained and churned customer groups, dynamically selecting the fairest mathematical test to compare them.
4. **The Redundancy Filter (Multicollinearity):** Sometimes features overlap so much that they tell the exact same story. For instance, "Total Charges Collected" is highly redundant if you already know a customer's "Monthly Bill Amount" and how many months they have stayed with the company. The pipeline calculates a relationship score for all features and automatically discards the overlapping ones to keep the model lightweight and stable.
5. **The Final Prediction Engine (Machine Learning):** Only the cleanest, most reliable, and completely independent customer features are allowed into the final machine learning classifier. The system trains a robust model that carefully handles imbalances (since remaining customers usually outnumber leaving customers) and outputs trustworthy predictions.

---

# 🏗️ Architecture & Module Blueprint

The codebase uses a highly decoupled, modular structure designed for enterprise stability:

````text
📦 telco-churn-statistical-pipeline/
├── 📂 config/
│   └── config.yaml              # Single source of truth for rules, metrics, and limits
├── 📂 src/
│   ├── __init__.py              # Exposes directory packages natively
│   ├── ingestion.py             # Raw file parsing, data type cleaning, and business logic fixes
│   ├── normality.py             # Shapiro-Wilk testing and distribution shape smoothing
│   ├── variance.py              # Levene / Bartlett variance consistency checks
│   ├── testing.py               # Automated feature engine selector (Mann-Whitney U routing)
│   ├── multicollinearity.py     # Iterative redundancy filtering via VIF
│   └── model.py                 # Stratified cross-validation, performance scoring, and asset serialization
├── 📄 app_copilot.py            # Streamlit multi-pane web application interface
├── 📄 mcp_server.py             # Model Context Protocol server exposing local ML inference tools
├── 📄 groq_agent.py             # Groq LLM orchestration and conversational memory handler
├── 📄 database_connection.py    # Persistent operational database linkage utilities
├── 📄 main.py                   # Master application orchestrator script for the ML pipeline
├── 📄 requirements.txt          # Fixed library dependencies snapshot
└── 📄 Dockerfile                # Container recipe for deployment isolation
'''
---
### ⚙️ Project File Registry

- **`config/config.yaml`**: Houses all decision boundaries, column maps, and target thresholds.
- **`src/ingestion.py`**: Performs type coercion and applies business-driven missing data imputation.
- **`src/normality.py`**: Runs Shapiro-Wilk data shape checks and Yeo-Johnson transformations.
- **`src/variance.py`**: Evaluates variance consistency using Levene or Bartlett tests.
- **`src/testing.py`**: Validates whether individual features are mathematically significant predictors.
- **`src/multicollinearity.py`**: Flags and discards columns with a VIF score greater than `5.0`.
- **`src/model.py`**: Executes Stratified 5-Fold cross-validation and trains the final classifier.
- **`app_copilot.py`**: Renders the high-density Streamlit user interface featuring a split-panel layout (Natural Language Chat Interface alongside the Background Model Execution Engine).
- **`mcp_server.py`**: Acts as the local tool server (`predict_customer_churn_risk` and `get_pipeline_performance_metadata`) called dynamically by the LLM agent.
- **`groq_agent.py`**: Configures API integration with high-performance Groq models (`qwen/qwen3.8-27b`).
# Pipeline Phase 1: Inference-Guided Statistical Gatekeeper

This document defines the mathematical preconditions, distribution shaping rules, and hypothesis testing infrastructure that acts as the automated gatekeeper for the analytical dataset.

---

## 1. Data Ingestion & Business Logic Imputation
Before running any mathematical calculations, raw data must be structurally sound. Rather than blindly dropping rows with missing values (which introduces selection bias and deletes all brand-new customer accounts), the pipeline applies deterministic domain logic:
*   **Zero-Tenure Rule:** If a subscriber's `tenure` is `0` months, their accrued `TotalCharges` are automatically set to `0.0`.
*   **Target Standardization:** The target column values are coerced into clear computer bits (`Yes` → `1`, `No` → `0`) to prevent data-type shifting down the road.

---

## 2. Normality Assessment & Shape Transformation
Standard machine learning models assume variables follow a natural, symmetric bell-shaped curve.
*   **The Check (Shapiro-Wilk Test):** Continuous variables are evaluated for normality. If the resulting p-value is less than our alpha threshold (α = 0.05), the feature is flagged as non-normal.
*   **The Fix (Yeo-Johnson Power Transform):** For variables with heavy tails or bimodal distributions (like `TotalCharges` and `tenure`), standard Z-score scaling is insufficient because it preserves asymmetry. The pipeline automatically routes non-normal features through a non-linear Yeo-Johnson transformation to smooth the data distribution before applying Standard Normal Variate (Z-score) standardization.

---

## 3. Equal Variance (Homoscedasticity) Testing
To compare customer groups fairly, the pipeline checks if data fluctuations remain steady across your categories (`Churn=0` vs `Churn=1`).
*   **Dynamic Test Routing:** If a feature is perfectly normal, it routes to **Bartlett's Test**. Since our features are non-normal and heavily skewed, the pipeline automatically routes them to **Levene’s Test (centered on the median)**. Centering on the median ensures the test remains robust and prevents asymmetric shapes from inflating false-positive rates.

---

## 4. Automated Hypothesis Testing (Feature Selection)
Instead of guessing which variables are useful, the pipeline uses the accumulated normality and variance logs to dynamically deploy the exact hypothesis test required to prove predictive significance:
*   **The Routing Engine:** Since the features exhibit non-normal distributions, the pipeline runs a non-parametric **Mann-Whitney U Test** to measure group differences across churn states.
*   **Significance Gate:** Features are rejected from the final training matrix if their computed p-value fails to clear the significance gate (p ≥ 0.05).

---

## 5. Multicollinearity Elimination (VIF Loop)
While hypothesis tests evaluate features one by one against the target, they do not check if variables overlap with *each other*.
*   **The Iterative VIF Loop:** The pipeline processes the continuous data matrix through an iterative Variance Inflation Factor (VIF) loop. If a feature registers an overlapping score higher than our structural limit (**VIF > 5.0**), it is flagged and discarded.
*   **The Result:** The loop successfully flags **`TotalCharges`** with a massive VIF of **48.12** and drops it. This stabilizes the remaining weights for `tenure` and `MonthlyCharges` down to a perfectly independent **1.06**.

# Pipeline Phase 2: Predictive Machine Learning Core

This document details the cross-validation architecture, dataset encoding schemas, and serialization layers used to compile stable, production-ready predictive models.

---

## 1. Feature Engineering & Matrix Encoding
Once continuous inputs are cleaned and filtered by the statistical gatekeeper, categorical parameters are incorporated to maximize predictive accuracy:
*   **One-Hot Encoding:** Categorical variables (such as `Contract` and `InternetService`) are mapped into a binary boolean matrix.
*   **Structural Alignment:** The continuous features and dummy-encoded vectors are concatenated into a final training matrix containing exactly **29 predictive features**.

---

## 2. Cross-Validation Architecture
To ensure performance metrics are steady and account for class imbalances (since remaining customers outnumber churned accounts), the system enforces strict evaluation rules:
*   **Stratified 5-Fold Cross-Validation:** The dataset is split into five distinct validation groups, preserving the exact target class ratio across every fold to prevent biased scoring.
*   **Classifier Engine:** The pipeline trains a **Balanced Random Forest Classifier**, utilizing class-weight adjustments to natively manage the underlying data skew.

---

## 3. Production Performance Report
The cross-validated validation framework returns highly stable, production-grade classification metrics:
*   **Mean Accuracy Rate:** `0.7843`
*   **Weighted F1-Score:** `0.7740`
*   **Mean ROC-AUC Score:** `0.8150`

### Core Structural Drivers (Feature Importances):
1.  **tenure** (Account Age in Months) — `22.92%` System Importance *(Validates our statistical phase where tenure registered the largest practical Cohen's d effect size of 0.9067)*
2.  **MonthlyCharges** (Current Billing Rate) — `20.20%` System Importance
3.  **Contract_Two year** (Long-Term Commitment Hold) — `6.49%` System Importance
4.  **InternetService_Fiber optic** (High-Speed Fiber Infrastructure) — `4.81%` System Importance
5.  **PaymentMethod_Electronic check** (Manual Monthly Billing Choice) — `3.95%` System Importance

---

## 4. Model Persistence & Serialization
When local code verification completes successfully via `python main.py`, the script freezes the states of the pipeline components and exports them into a dedicated `artifacts/` folder as binary **`.pkl`** (pickle) files:
*   `churn_classifier_model.pkl`: The fully trained production random forest model.
*   `production_feature_columns.pkl`: The frozen list of 29 features to guarantee strict column alignment during real-time inference.
*   `normality_transformers.pkl`: The transformation logs and scaling offsets.

# Pipeline Phase 3: Conversational GenAI Co-Pilot Hub

This document serves as the operational quick-start guide and deployment manual for the interactive split-pane web application interface.

---

## 1. Application Interface Overview
The **`app_copilot.py`** module builds a single-page web dashboard that combines conversational AI with real-time background analytics. The application is divided into a clean, high-density split-pane layout:
*   **Left Pane (Natural Language Interface):** An interactive chat panel running the **Qwen 3.8 27B** model via Groq. Users can ask general knowledge questions or describe a subscriber's profile in plain English.
*   **Right Pane (Background Execution Engine):** Displays a live, synchronized visual view of the model's metrics. When a customer profile prompt is typed on the left, the app runs local machine learning calculations in the background, updates a visual progress gauge, and displays risk tier badges instantly on the right.

---

## 2. Interactive Tool Integration (MCP Framework)
The application handles user messages through a smart tool execution routing loop. It scans user input for keywords like "predict", "risk", or "churn" and bridges the text directly to our local python model functions:
*   **Live Transformation Layer:** The app dynamically reads input parameters, applies baseline scaling approximations to the continuous variables, dummy-encodes categorical values, and maps the results to the exact 29-column matrix our model expects.
*   **The Response Loop:** Once the local model calculates the probability, the data is fed back to the Qwen agent. The LLM translates the raw percentage into a clear, plain-English executive summary and lays out a context-aware strategic retention playbook.

---

## 3. Operational Risk Tier Segments
The conversational co-pilot segments results into three clear enterprise risk boundaries:
1.  **Stable (Low Risk):** Churn probability under 30%. The app displays a green status banner and advises maintaining standard automated marketing workflows.
2.  **Elevated (Moderate Risk):** Churn probability between 30% and 49.99%. The app displays a yellow status banner and recommends soft retention actions, like offering minor billing credits.
3.  **Critical (High Risk):** Churn probability of 50% and above. The app displays a red status warning and lays out immediate account recovery triggers, like contract optimizations or outbound care calls.

---

## 4. Quick-Start Execution Guide

### Local Launch In Instructions
1.  Ensure your Groq key is written inside a secure **`.env`** file in your root folder:
    ```env
    GROQ_API_KEY=gsk_your_actual_secret_key_here
    ```
2.  Launch the conversational dashboard from your terminal window:
    ```bash
    streamlit run app_copilot.py
    ```

### Containerized Deployment via Docker
To isolate the application completely and ensure it runs identically across any environment, build and launch the container image using folder volume mapping (`-v` parameter) to share the raw data file:
```bash
docker build -t statistical-ml-pipeline:latest .
docker run --rm -v "%cd%:/app" statistical-ml-pipeline:latest
````

_(Note: If executing this step from a Windows PowerShell terminal instead of a classic Command Prompt, use `-v "${PWD}:/app"` instead)._
