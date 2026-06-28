import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

import utils
from preprocessing import preprocess_data
from predict import FraudPredictor
from train_model import train_and_evaluate_all

# Set Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
    }
    
    /* Metrics Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }
    
    /* Risk Levels styling */
    .risk-low {
        color: #2ECC71;
        font-weight: bold;
    }
    .risk-medium {
        color: #F39C12;
        font-weight: bold;
    }
    .risk-high {
        color: #E74C3C;
        font-weight: bold;
    }
    
    /* Header Accent */
    .header-container {
        border-bottom: 2px solid #1E293B;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }
    
    /* Streamlit overrides */
    div[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if models are trained
def check_models_exist():
    return (
        os.path.exists(os.path.join(utils.MODELS_DIR, "best_model.joblib")) and
        os.path.exists(os.path.join(utils.MODELS_DIR, "training_metadata.joblib")) and
        os.path.exists(os.path.join(utils.MODELS_DIR, "scaler.joblib"))
    )

# Cache data loading
@st.cache_data(show_spinner="Loading Kaggle Dataset (150MB)...")
def load_cached_data():
    csv_path = utils.download_dataset()
    col_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.read_csv(csv_path, header=None, names=col_names)
    return df

@st.cache_data
def get_correlation_matrix(df):
    # Select numeric features
    cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class']
    corr = df[cols].corr()
    return corr, cols

# ----------------- MAIN FLOW ----------------- #

# Title
st.markdown("""
<div class="header-container">
    <h1 style='margin: 0; font-weight: 800; font-size: 2.5rem; background: linear-gradient(to right, #60A5FA, #3B82F6, #1D4ED8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🛡️ Credit Card Fraud Detection Hub
    </h1>
    <p style='margin: 5px 0 0 0; color: #94A3B8; font-size: 1.1rem;'>
        An end-to-end Machine Learning suite detecting transaction anomalies in real-time.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. Verification of models
if not check_models_exist():
    st.info("👋 **Welcome to the Fraud Detection Hub!**")
    st.warning("No pre-trained model files were found on this system. We need to train the models first to initialize the application.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 Train Models & Initialize Hub", use_container_width=True, type="primary"):
            with st.spinner("Preparing dataset, applying SMOTE, and training 4 classifiers... This may take up to a minute."):
                try:
                    # Run training pipeline
                    train_and_evaluate_all()
                    st.success("Models trained successfully!")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
    with col2:
        st.markdown("""
        **What will happen when you click train?**
        1. **Download Kaggle Data:** Downloads the creditcard dataset (~66MB zip) from GitHub.
        2. **Clean & Scale:** Automatically scales Time & Amount and removes training duplicates.
        3. **SMOTE Oversampling:** Adjusts class imbalance (only 0.17% fraud) dynamically.
        4. **Compare Models:** Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost.
        5. **Save Artifacts:** Saves all classifiers, scalers, and metric comparison details.
        """)
    st.stop()

# Load metadata & cached dataset
metadata = utils.load_artifact("training_metadata.joblib")
best_model_name = metadata.get('best_model_name', 'Random Forest')
model_metrics = metadata.get('metrics', {})
roc_curves = metadata.get('roc_curves', {})

df = load_cached_data()

# ----------------- SIDEBAR ----------------- #
st.sidebar.markdown("<h2 style='text-align: center; color: #3B82F6;'>Settings</h2>", unsafe_allow_html=True)

# Model selection override
model_options = {
    "Best Model (Auto Selected)": "best_model.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib"
}
selected_model_key = st.sidebar.selectbox("Active Classifier Model", list(model_options.keys()))
selected_model_file = model_options[selected_model_key]

# Status box
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")

# Render active model tag
active_name = best_model_name if selected_model_key == "Best Model (Auto Selected)" else selected_model_key
st.sidebar.info(f"**Active Classifier:**\n{active_name}")

# Quick info
st.sidebar.markdown(f"""
- **Dataset Size:** {len(df):,} transactions
- **Imbalance Ratio:** {df['Class'].mean()*100:.3f}% Fraud
- **Original Scale:** V1-V28 (PCA), Time (s), Amount ($)
""")

# Retraining Action
st.sidebar.markdown("---")
st.sidebar.markdown("### System Maintenance")
if st.sidebar.button("🔄 Retrain All Models", use_container_width=True):
    with st.spinner("Retraining pipeline..."):
        try:
            train_and_evaluate_all()
            st.sidebar.success("Retrained successfully!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Retraining failed: {str(e)}")

# Initialize Predictor
try:
    predictor = FraudPredictor(model_name=selected_model_file)
except Exception as e:
    st.error(f"Failed to load predictor: {str(e)}")
    st.stop()

# ----------------- MAIN TABS ----------------- #
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & Analytics", 
    "📈 Model Evaluation", 
    "🔍 Real-time Predictor", 
    "📁 Batch Prediction"
])

# ================= TAB 1: OVERVIEW & EDA ================= #
with tab1:
    st.markdown("### Transaction Dataset Metrics")
    
    total_tx = len(df)
    total_fraud = (df['Class'] == 1).sum()
    fraud_pct = (total_fraud / total_tx) * 100
    avg_amt = df['Amount'].mean()
    
    # Grid metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 14px; font-weight: 600;">Total Transactions</div>
            <div class="metric-value" style="color: #60A5FA;">{total_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 14px; font-weight: 600;">Fraud Transactions</div>
            <div class="metric-value" style="color: #F87171;">{total_fraud:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 14px; font-weight: 600;">Fraud Rate (%)</div>
            <div class="metric-value" style="color: #FBBF24;">{fraud_pct:.4f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 14px; font-weight: 600;">Avg Amount ($)</div>
            <div class="metric-value" style="color: #34D399;">${avg_amt:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Class Distribution & Hourly Distribution
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.plotly_chart(utils.plot_class_distribution(df['Class']), use_container_width=True)
    with col_dist2:
        st.plotly_chart(utils.plot_time_distribution(df), use_container_width=True)
        
    st.markdown("---")
    
    # Transaction Amount boxplot & Correlation
    col_plot1, col_plot2 = st.columns([1, 1])
    with col_plot1:
        st.plotly_chart(utils.plot_transaction_amount_distribution(df), use_container_width=True)
    with col_plot2:
        corr_matrix, feature_names = get_correlation_matrix(df)
        st.plotly_chart(utils.plot_correlation_heatmap(corr_matrix, feature_names), use_container_width=True)


# ================= TAB 2: MODEL EVALUATION ================= #
with tab2:
    st.markdown("### Machine Learning Model Comparison")
    st.markdown("All models were trained on oversampled data (SMOTE) and evaluated on a clean, stratified 20% holdout test set.")
    
    # Create comparison table
    df_metrics = pd.DataFrame(model_metrics).T
    df_display = df_metrics[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']]
    
    # Highlight best in table
    st.dataframe(
        df_display.style.highlight_max(subset=['Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color="#1E3A8A")
                         .format(precision=4),
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bar Chart for specific metric comparison
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        metric_choice = st.selectbox("Select metric to compare:", ['F1-Score', 'Recall', 'Precision', 'Accuracy', 'ROC-AUC'])
        df_sorted = df_display.sort_values(by=metric_choice, ascending=False)
        fig_metric = px.bar(
            df_sorted,
            x=df_sorted.index,
            y=metric_choice,
            color=metric_choice,
            color_continuous_scale='Viridis',
            title=f"Models Ranked by {metric_choice}"
        )
        fig_metric.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_metric, use_container_width=True)
        
    with col_comp2:
        st.plotly_chart(utils.plot_roc_curve(roc_curves), use_container_width=True)
        
    st.markdown("---")
    
    # Confusion Matrix & Feature Importance
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        # Load CM for currently active model
        active_metrics = model_metrics.get(active_name, None)
        if active_metrics and 'Confusion Matrix' in active_metrics:
            cm = np.array(active_metrics['Confusion Matrix'])
            st.plotly_chart(utils.plot_confusion_matrix(cm, active_name), use_container_width=True)
        else:
            st.info("No confusion matrix available for this model.")
            
    with col_cm2:
        # Display feature importances of the selected model if available
        active_model = predictor.model
        features = predictor.expected_features
        
        importances = None
        if hasattr(active_model, "feature_importances_"):
            importances = active_model.feature_importances_
        elif hasattr(active_model, "coef_"):
            importances = np.abs(active_model.coef_[0])
            
        if importances is not None:
            st.plotly_chart(utils.plot_feature_importance(features, importances, active_name), use_container_width=True)
        else:
            st.info(f"Feature importance is not directly supported by {active_name}.")


# ================= TAB 3: REAL-TIME PREDICTOR ================= #
with tab3:
    st.markdown("### Manual Transaction Testing")
    st.markdown("Load a real transaction sample from the test set or enter manual attributes to predict the likelihood of fraud.")
    
    # Load presets
    presets = utils.load_artifact("test_presets.joblib")
    
    preset_choice = "Custom Entry"
    if presets is not None:
        preset_names = ["Custom Entry", "Zero Initialized"]
        preset_names.extend([f"Legitimate Case #{i+1}" for i in range(10)])
        preset_names.extend([f"Fraudulent Case #{i+1}" for i in range(10)])
        
        preset_choice = st.selectbox("🧪 Load Data Preset (Autofill Form)", preset_names)
        
    # Initialize values
    input_values = {}
    
    if preset_choice == "Zero Initialized":
        for col in predictor.expected_features:
            input_values[col] = 0.0
    elif preset_choice.startswith("Legitimate Case"):
        idx = int(preset_choice.split("#")[1]) - 1
        legit_df = presets[presets['Class'] == 0].reset_index(drop=True)
        row = legit_df.iloc[idx]
        for col in predictor.expected_features:
            input_values[col] = float(row[col])
    elif preset_choice.startswith("Fraudulent Case"):
        idx = int(preset_choice.split("#")[1]) - 1
        fraud_df = presets[presets['Class'] == 1].reset_index(drop=True)
        row = fraud_df.iloc[idx]
        for col in predictor.expected_features:
            input_values[col] = float(row[col])
    else: # Custom Entry
        # Default V1-V28 to 0, Time to 1000, Amount to 50
        for col in predictor.expected_features:
            if col == 'Time':
                input_values[col] = 86400.0 # 1 day
            elif col == 'Amount':
                input_values[col] = 100.0
            else:
                input_values[col] = 0.0

    # User Input Form structured as neat grids
    st.markdown("#### Transaction Parameters")
    with st.form("single_prediction_form"):
        # Time & Amount (high level)
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            input_values['Time'] = st.number_input(
                "Transaction Time (seconds from start)", 
                value=float(input_values['Time']),
                help="Number of seconds elapsed between this transaction and the first transaction in the dataset."
            )
        with c_t2:
            input_values['Amount'] = st.number_input(
                "Transaction Amount ($)", 
                min_value=0.0, 
                value=float(input_values['Amount']),
                help="Amount of the transaction in USD."
            )
            
        st.markdown("<h5 style='margin-bottom: 5px;'>PCA Anonymized Features (V1 to V28)</h5>", unsafe_allow_html=True)
        
        # Grid of V1-V28 columns
        v_cols = [f"V{i}" for i in range(1, 29)]
        grid_cols = st.columns(4) # 4 columns
        
        for i, col_name in enumerate(v_cols):
            col_idx = i % 4
            with grid_cols[col_idx]:
                input_values[col_name] = st.number_input(
                    col_name,
                    value=float(input_values[col_name]),
                    format="%.6f"
                )
                
        submit_btn = st.form_submit_button("🔍 Analyze Transaction", type="primary", use_container_width=True)
        
    if submit_btn:
        with st.spinner("Analyzing transaction patterns..."):
            try:
                # Predict
                res = predictor.predict_single(input_values)
                prob = res['Fraud_Probability']
                pred = res['Fraud_Prediction']
                conf = res['Confidence']
                risk = res['Risk_Level']
                
                st.markdown("### Prediction Results")
                
                # Double column output
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    # Guage indicator
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability Gauge", 'font': {'size': 18, 'family': 'Inter, sans-serif'}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                            'bar': {'color': "#ffffff"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "#334155",
                            'steps': [
                                {'range': [0, 20], 'color': '#2ECC71'},
                                {'range': [20, 70], 'color': '#F39C12'},
                                {'range': [70, 100], 'color': '#E74C3C'}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': prob * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=280,
                        margin=dict(t=30, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with col_res2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    # Status alert banner
                    if pred == 1:
                        st.markdown(f"""
                        <div style="background-color: rgba(231, 76, 60, 0.15); border: 2px solid #E74C3C; border-radius: 8px; padding: 20px;">
                            <h3 style="color: #E74C3C; margin: 0 0 10px 0; font-weight: 800;">🛑 FRAUD DETECTED</h3>
                            <p style="margin: 0; font-size: 16px;">This transaction matches patterns typical of fraudulent behavior. The system recommends blocking this card activity immediately.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: rgba(46, 204, 113, 0.15); border: 2px solid #2ECC71; border-radius: 8px; padding: 20px;">
                            <h3 style="color: #2ECC71; margin: 0 0 10px 0; font-weight: 800;">✅ LEGITIMATE TRANSACTION</h3>
                            <p style="margin: 0; font-size: 16px;">The transaction profile is regular. No threat detected.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Extra details table
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Risk class mapping
                    risk_class = "risk-low"
                    if "Medium" in risk:
                        risk_class = "risk-medium"
                    elif "High" in risk:
                        risk_class = "risk-high"
                        
                    st.markdown(f"""
                    **Security Report Summary:**
                    * **Evaluation Model:** `{active_name}`
                    * **Target Prediction:** `{'Fraudulent' if pred == 1 else 'Legitimate'}`
                    * **Fraud Probability:** `{prob*100:.3f}%`
                    * **Model Confidence:** `{conf*100:.3f}%`
                    * **Assessed Risk Category:** <span class="{risk_class}">{risk}</span>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


# ================= TAB 4: BATCH PREDICTION ================= #
with tab4:
    st.markdown("### CSV Batch Processing")
    st.markdown("Upload a CSV file containing transactions to evaluate multiple records simultaneously. The CSV should contain the same 30 features (`Time`, `Amount`, and `V1` to `V28`).")
    
    # Template download helper
    test_presets = utils.load_artifact("test_presets.joblib")
    if test_presets is not None:
        template_df = test_presets.drop('Class', axis=1, errors='ignore').head(5)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Batch Template",
            data=csv_template,
            file_name="fraud_batch_template.csv",
            mime="text/csv",
            help="Download a 5-row CSV template with correct headers."
        )
        
    uploaded_file = st.file_uploader("Upload your batch transaction CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file: **{uploaded_file.name}** ({len(df_batch)} rows)")
            
            # Show preview
            st.markdown("#### Input Data Preview (Top 5 rows)")
            st.dataframe(df_batch.head(5), use_container_width=True)
            
            # Predict Button
            if st.button("⚡ Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing batch dataset..."):
                    df_results = predictor.predict_batch(df_batch)
                    
                    # Highlight and output
                    st.markdown("---")
                    st.markdown("### Batch Prediction Results")
                    
                    # Summary metrics
                    total_records = len(df_results)
                    detected_fraud = (df_results['Fraud_Prediction'] == 1).sum()
                    detected_legit = total_records - detected_fraud
                    detected_fraud_pct = (detected_fraud / total_records) * 100
                    
                    col_batch1, col_batch2 = st.columns([1, 2])
                    
                    with col_batch1:
                        # Pie chart for batch prediction split
                        fig_batch_pie = go.Figure(data=[go.Pie(
                            labels=['Legitimate', 'Fraud'],
                            values=[detected_legit, detected_fraud],
                            hole=0.4,
                            marker=dict(colors=['#2ECC71', '#E74C3C']),
                            textinfo='label+percent'
                        )])
                        fig_batch_pie.update_layout(
                            title="Batch Predictions Split",
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=55, b=10, l=10, r=10),
                            height=250
                        )
                        st.plotly_chart(fig_batch_pie, use_container_width=True)
                        
                    with col_batch2:
                        st.markdown(f"""
                        **Batch Execution Summary:**
                        * **Total Transactions Evaluated:** `{total_records:,}`
                        * **Fraud Flags Triggered:** `{detected_fraud:,} ({detected_fraud_pct:.3f}%)`
                        * **Legitimate Claims Cleared:** `{detected_legit:,} ({100 - detected_fraud_pct:.3f}%)`
                        * **Detection Model Used:** `{active_name}`
                        """)
                        
                    # Results Table Preview
                    st.markdown("#### Predictions Table (First 15 Rows)")
                    cols_to_show = ['Fraud_Prediction', 'Fraud_Probability', 'Confidence', 'Risk_Level', 'Time', 'Amount']
                    other_cols = [c for c in df_results.columns if c not in cols_to_show]
                    ordered_cols = cols_to_show + other_cols
                    
                    # Style results
                    st.dataframe(
                        df_results[ordered_cols].head(15).style.map(
                            lambda val: 'background-color: rgba(231, 76, 60, 0.2);' if val == 'High Risk (Potential Fraud)' else '', 
                            subset=['Risk_Level']
                        ),
                        use_container_width=True
                    )
                    
                    # Download Prediction Button
                    csv_results = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Full Predictions CSV",
                        data=csv_results,
                        file_name="transaction_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"Error parsing or predicting batch file: {str(e)}")
            st.exception(e)
