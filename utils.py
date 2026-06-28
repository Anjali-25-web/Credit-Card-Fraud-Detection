import os
import urllib.request
import zipfile
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/creditcard.csv.zip"
ZIP_PATH = os.path.join(DATA_DIR, "creditcard.csv.zip")
CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")

def ensure_directories():
    """Ensure that data and models directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def download_dataset():
    """
    Downloads the creditcard.csv.zip dataset from jbrownlee/Datasets GitHub mirror,
    unzips it in the data directory, and removes the zip file.
    """
    ensure_directories()
    
    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists at: {CSV_PATH}")
        return CSV_PATH
    
    print(f"Dataset not found. Downloading from {DATASET_URL}...")
    try:
        # User-agent header to avoid potential rate limiting
        req = urllib.request.Request(
            DATASET_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(ZIP_PATH, 'wb') as out_file:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            print(f"Content-Length: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
            
            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
                if file_size:
                    percent = (downloaded / file_size) * 100
                    print(f"Progress: {downloaded / (1024*1024):.1f} MB / {file_size / (1024*1024):.1f} MB ({percent:.1f}%)", flush=True)
                else:
                    print(f"Progress: {downloaded / (1024*1024):.1f} MB downloaded", flush=True)
            print("Download finished.")
            
        print("Download complete. Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        # Clean up zip file
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            
        print(f"Dataset successfully downloaded and extracted to: {CSV_PATH}")
        return CSV_PATH
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        # If download fails, check if zip or csv exists locally by chance
        if os.path.exists(CSV_PATH):
            return CSV_PATH
        raise e

def save_artifact(obj, filename):
    """
    Saves an object (model, scaler, etc.) to the models directory.
    
    Args:
        obj: The object to save.
        filename (str): The filename for the saved artifact.
    """
    ensure_directories()
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, filepath)
    print(f"Artifact successfully saved to: {filepath}")

def load_artifact(filename):
    """
    Loads an object (model, scaler, etc.) from the models directory.
    
    Args:
        filename (str): The filename of the saved artifact.
        
    Returns:
        The loaded object, or None if the file doesn't exist.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Artifact not found at: {filepath}")
        return None
    return joblib.load(filepath)

# ----------------- Visualizations Helpers ----------------- #

def plot_class_distribution(y):
    """
    Generates a Plotly Pie chart showing the class distribution of transactions.
    
    Args:
        y: Series or array of target labels (0 = Legitimate, 1 = Fraud).
        
    Returns:
        plotly.graph_objects.Figure: The pie chart figure.
    """
    counts = pd.Series(y).value_counts()
    labels = ['Legitimate', 'Fraud']
    values = [counts.get(0, 0), counts.get(1, 0)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#2ECC71', '#E74C3C']),
        textinfo='label+percent',
        insidetextorientation='radial',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title="Transaction Class Distribution",
        title_font=dict(size=18, family="Inter, sans-serif"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=10, l=10, r=10)
    )
    return fig

def plot_confusion_matrix(cm, model_name="Model"):
    """
    Generates a Plotly heatmap representing the confusion matrix.
    
    Args:
        cm: Confusion matrix (2x2 numpy array).
        model_name (str): Name of the model.
        
    Returns:
        plotly.graph_objects.Figure: Confusion matrix heatmap.
    """
    # Normalized confusion matrix for color scaling
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Annotations
    z_text = [[f"True Neg:<br>{cm[0][0]}", f"False Pos:<br>{cm[0][1]}"],
              [f"False Neg:<br>{cm[1][0]}", f"True Pos:<br>{cm[1][1]}"]]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=['Predicted Legitimate', 'Predicted Fraud'],
        y=['Actual Legitimate', 'Actual Fraud'],
        text=z_text,
        texttemplate="%{text}",
        hoverinfo="z",
        colorscale=[[0.0, '#1E293B'], [0.5, '#0F172A'], [1.0, '#E74C3C']],
        showscale=False
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        title_font=dict(size=18, family="Inter, sans-serif"),
        xaxis=dict(side="bottom"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50, l=50, r=50),
        height=400
    )
    return fig

def plot_roc_curve(models_evaluation_data):
    """
    Generates a Plotly line chart comparing ROC Curves for multiple models.
    
    Args:
        models_evaluation_data: Dict with model names as keys and a dict containing
                                'fpr', 'tpr', and 'roc_auc' as values.
                                
    Returns:
        plotly.graph_objects.Figure: ROC Curve figure.
    """
    fig = go.Figure()
    
    # Baseline random model
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.50)',
        line=dict(dash='dash', color='#64748B')
    ))
    
    # Add each model's ROC Curve
    for name, data in models_evaluation_data.items():
        if 'fpr' in data and 'tpr' in data:
            fig.add_trace(go.Scatter(
                x=data['fpr'], y=data['tpr'],
                mode='lines',
                name=f"{name} (AUC = {data['roc_auc']:.4f})",
                line=dict(width=2)
            ))
            
    fig.update_layout(
        title="ROC Curves Comparison",
        title_font=dict(size=18, family="Inter, sans-serif"),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95, bgcolor="rgba(15, 23, 42, 0.8)"),
        margin=dict(t=50, b=50, l=50, r=50),
        height=500
    )
    return fig

def plot_feature_importance(feature_names, importances, model_name="Model", max_features=15):
    """
    Generates a Plotly bar chart showing feature importances.
    
    Args:
        feature_names: List of feature names.
        importances: Array of importance weights.
        model_name (str): Name of the model.
        max_features (int): Maximum number of features to show.
        
    Returns:
        plotly.graph_objects.Figure: Horizontal bar chart figure.
    """
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Get top max_features
    df_importance = df_importance.tail(max_features)
    
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {max_features} Feature Importances - {model_name}",
        color='Importance',
        color_continuous_scale='Bluered'
    )
    
    fig.update_layout(
        title_font=dict(size=18, family="Inter, sans-serif"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(t=50, b=55, l=10, r=10),
        height=500
    )
    return fig

def plot_correlation_heatmap(corr_matrix, feature_names):
    """
    Generates a Plotly heatmap representing feature correlation.
    
    Args:
        corr_matrix: Correlation matrix DataFrame or numpy array.
        feature_names: List of feature names.
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap figure.
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=feature_names,
        y=feature_names,
        colorscale='RdBu_r', # Red for positive correlation, Blue for negative
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        title_font=dict(size=18, family="Inter, sans-serif"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50, l=50, r=50),
        height=600
    )
    return fig

def plot_transaction_amount_distribution(df):
    """
    Generates an interactive overlay plot showing distributions of transaction amounts for fraud vs legitimate.
    
    Args:
        df: DataFrame containing 'Amount' and 'Class' columns.
        
    Returns:
        plotly.graph_objects.Figure: Distribution figure.
    """
    # Box plot of Transaction Amount by Class
    fig = px.box(
        df,
        x='Class',
        y='Amount',
        color='Class',
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        category_orders={"Class": [0, 1]},
        labels={'Class': 'Transaction Type', 'Amount': 'Transaction Amount ($)'},
        title="Transaction Amount Distribution by Class (Log Scale)"
    )
    
    # Update category names for visualization
    fig.update_xaxes(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
    fig.update_yaxes(type="log") # Use log scale since amounts can have massive outliers
    
    fig.update_layout(
        title_font=dict(size=18, family="Inter, sans-serif"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50),
        height=450
    )
    return fig

def plot_time_distribution(df):
    """
    Generates histograms comparing transaction hour distribution between Class 0 and Class 1.
    
    Args:
        df: DataFrame containing 'Time' and 'Class' columns.
        
    Returns:
        plotly.graph_objects.Figure: Hour density distribution figure.
    """
    # Convert Time (seconds from start) to Hour of Day (0-23)
    df_time = df.copy()
    df_time['Hour'] = (df_time['Time'] / 3600) % 24
    
    fig = go.Figure()
    
    # Legitimate transactions distribution
    fig.add_trace(go.Histogram(
        x=df_time[df_time['Class'] == 0]['Hour'],
        name='Legitimate',
        marker_color='#2ECC71',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=24
    ))
    
    # Fraudulent transactions distribution
    fig.add_trace(go.Histogram(
        x=df_time[df_time['Class'] == 1]['Hour'],
        name='Fraud',
        marker_color='#E74C3C',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=24
    ))
    
    fig.update_layout(
        title="Transaction Hourly Density (00:00 to 23:59)",
        title_font=dict(size=18, family="Inter, sans-serif"),
        xaxis_title="Hour of Day",
        yaxis_title="Probability Density",
        barmode='overlay',
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50, l=50, r=50),
        height=450,
        legend=dict(bgcolor="rgba(15, 23, 42, 0.8)")
    )
    return fig
