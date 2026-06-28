# Credit Card Fraud Detection System

An end-to-end machine learning system designed to detect fraudulent credit card transactions and analyze transaction behavior using an interactive Streamlit dashboard. 

The system leverages anonymized PCA-transformed features (V1-V28), transaction timestamps, and transaction amounts from the classic Kaggle Credit Card Fraud dataset. It automatically handles extreme class imbalance using SMOTE and compares multiple machine learning classifiers to select the best performer.

---

## 🌟 Key Features

*   **Robust Data Preprocessing:**
    *   Automatic missing value checking and column-type validation.
    *   Oversampling via **SMOTE** (Synthetic Minority Over-sampling Technique) applied strictly to training splits.
    *   Feature scaling of `Time` and `Amount` using **RobustScaler** to handle extreme transaction outliers.
    *   Duplicate record removal.
*   **Multi-Model Training & Evaluation:**
    *   Trains and evaluates four ML classifiers: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**.
    *   Evaluates models using standard classification metrics: **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
    *   Selects and saves the best model based on the **F1-Score** (optimal metric for balanced precision/recall on imbalanced datasets).
*   **Professional Streamlit Dashboard:**
    *   **Overview & EDA Tab:** High-level system metrics (total transactions, fraud rate), interactive distribution graphs, and feature correlation heatmaps.
    *   **Model Evaluation Tab:** Comparative grid of model metrics, dynamic ranking charts, interactive confusion matrices, and feature importance visualizers.
    *   **Real-time Predictor Tab:** Live prediction form with 30 parameters, integrated with a **Preset Loader** (load actual test cases with a click) and a visual **Fraud Probability Gauge**.
    *   **Batch Prediction Tab:** Drag-and-drop CSV batch transaction upload, live processing, and bulk output download as a CSV file.

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│── app.py                 # Streamlit dashboard application
│── train_model.py         # Model training, comparison, and selection pipeline
│── predict.py             # Inference class and CLI prediction interface
│── preprocessing.py       # Data cleaning, scaling, and SMOTE implementation
│── utils.py               # Data downlader, artifact saver, Plotly charting helpers
│── requirements.txt       # Project python dependencies
│── README.md              # Project documentation
│── models/                # Directory containing trained models and scalers (generated)
│── data/                  # Directory containing the downloaded dataset (generated)
│── assets/                # Visual assets and screenshots
```

---

## 🛠️ Tech Stack

*   **Programming Language:** Python
*   **Web Framework:** Streamlit
*   **Data Analysis:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
*   **Visualization:** Plotly, Matplotlib
*   **Model Storage:** Joblib

---

## 🚀 Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone or Move to the Directory
Open your terminal and navigate to the project directory:
```bash
cd "C:\Users\Anjali\Desktop\Credit Card Fraud Detection"
```

### 2. Create and Activate a Virtual Environment (Recommended)
Creating a virtual environment ensures dependencies do not conflict with your global python setup:
```bash
# Create environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Step 1: Train the Models
To initialize the models and prepare the metrics, execute the training script. This script will download the zip archive of the dataset (~66MB), unzip it, preprocess the data, run SMOTE, train all models, and select the best classifier:
```bash
python train_model.py
```
*Note: If you skip this step, you can trigger model training directly inside the Streamlit dashboard on startup!*

### Step 2: Launch the Streamlit Dashboard
Run the following command to launch the dashboard in your web browser:
```bash
streamlit run app.py
```
Once run, the terminal will output a local network URL (typically `http://localhost:8501`). Copy this URL into your browser to start exploring the system.

### Step 3: Run Predictions via CLI (Optional)
You can also use the CLI utility to run prediction tasks directly from the terminal.
*   **Single Transaction Prediction (JSON Input):**
    ```bash
    python predict.py --input "{\"Time\": 86400, \"V1\": -1.35, \"V2\": 1.21, ..., \"Amount\": 15.0}"
    ```
*   **Batch Prediction (CSV Input):**
    ```bash
    python predict.py --input "data/my_transactions.csv" --output "data/predictions_output.csv"
    ```

---

## 📊 Evaluation Metrics Overview

Since credit card fraud datasets are highly imbalanced (legitimate cases outnumber fraud cases ~578 to 1), traditional **Accuracy** is misleading. A model predicting "Legitimate" for all transactions would achieve 99.82% accuracy but detect 0% of fraud.

Thus, this project focuses on:
*   **Recall:** Maximizes the proportion of actual fraud cases caught (critical to limit financial losses).
*   **Precision:** Minimizes false fraud alerts (critical to maintain user trust and avoid card declines).
*   **F1-Score:** The harmonic mean of precision and recall, representing the primary optimization target.

---

## 🔮 Future Enhancements

*   **Feature Engineering:** Adding aggregate rolling features (e.g., *cumulative transaction volume over the last 2 hours*, *frequency of card usage*) to capture temporal behaviors.
*   **Deep Learning Models:** Integrating artificial neural networks (ANNs) or Autoencoders for semi-supervised anomaly detection.
*   **Database Integration:** Connecting the backend to PostgreSQL or MongoDB to fetch and write transaction records in real-time.
*   **API Deployment:** Wrapping the `predict.py` predictor class inside a FastAPI web service, allowing external client applications to make HTTP prediction queries.
