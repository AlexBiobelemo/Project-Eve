import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
import io
import json
import openpyxl
import numpy as np
from datetime import datetime
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import base64
from contextlib import contextmanager

# --- Constants ---
FILE_TYPES = ["CSV", "Excel", "JSON"]
CHART_OPTIONS = ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Correlation Heatmap", "Pie Chart"]
ALGORITHM_OPTIONS = ["K-Means Clustering", "DBSCAN Clustering", "Random Forest", "XGBoost"]
MISSING_VALUE_OPTIONS = ["Fill with Mean", "Drop Rows"]
THEME_OPTIONS = ["plotly", "plotly_dark", "seaborn"]
COLOR_PALETTES = ["Viridis", "Plasma", "Inferno", "Magma"]
LEGEND_POSITIONS = ["top", "bottom", "left", "right", "none"]

# --- Setup Logging ---
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Session State ---
if 'chart_configs' not in st.session_state:
    st.session_state.chart_configs = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {}
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}
if 'selected_df' not in st.session_state:
    st.session_state.selected_df = None
if 'error_messages' not in st.session_state:
    st.session_state.error_messages = []

# --- Helper Functions ---

@contextmanager
def status_message(message: str):
    """Context manager for temporary status messages."""
    placeholder = st.empty()
    placeholder.info(message)
    yield
    placeholder.empty()

@st.cache_data(show_spinner=False)
def load_data(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile], file_type: str) -> Optional[pd.DataFrame]:
    """Loads data from various file formats."""
    with status_message("Loading file..."):
        try:
            uploaded_file.seek(0)
            if file_type == "CSV":
                for sep in [',', ';', '\t']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=sep)
                        if len(df.columns) > 1:
                            break
                    except:
                        pass
                else:
                    raise ValueError("Could not parse CSV. Try a different separator or check file format.")
            elif file_type == "Excel":
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_type == "JSON":
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError("Unsupported file type")
            df.columns = [str(col).strip().replace(' ', '_').replace(';', '_') for col in df.columns]
            df = df.reset_index(drop=True)
            return df
        except Exception as e:
            logging.error(f"Data loading error: {e}")
            st.session_state.error_messages.append(f"Failed to load file: {str(e)}. Ensure correct format and separator.")
            return None

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Converts DataFrame to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

def suggest_chart_type(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """Suggests chart types based on data."""
    if len(numeric_cols) >= 2:
        return ["Correlation Heatmap", "Scatter Plot", "Box Plot"]
    elif categorical_cols and numeric_cols:
        return ["Bar Chart", "Box Plot"]
    elif numeric_cols:
        return ["Histogram", "Box Plot"]
    return ["Table View"]

@st.cache_data
def compute_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates EDA summary."""
    if df.empty:
        return {'shape': (0, 0), 'describe': {}, 'dtypes': {}, 'missing': {}, 'insights': []}
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = df[numeric_cols].corr() if numeric_cols else pd.DataFrame()
    insights = []
    if not corr_matrix.empty:
        high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
        for col1 in high_corr.columns:
            for col2 in high_corr.index:
                if high_corr.loc[col2, col1] and col1 < col2:
                    insights.append(f"High correlation between {col1} and {col2} ({corr_matrix.loc[col2, col1]:.2f}).")
    if not insights:
        insights.append("No significant correlations found.")
    return {
        'shape': df.shape,
        'describe': df.describe(include='all').to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'insights': insights
    }

def preprocess_clustering(data: pd.DataFrame, numeric_cols: List[str], missing_value_strategy: str) -> pd.DataFrame:
    """Preprocesses data for clustering."""
    if not numeric_cols:
        raise ValueError("No numeric columns selected for clustering.")
    data = data[numeric_cols].copy()
    if missing_value_strategy == "Drop Rows":
        data = data.dropna()
        if len(data) == 0:
            raise ValueError("No data remains after dropping rows with missing values.")
    else:
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        if data[numeric_cols].isna().all().any():
            raise ValueError(f"Columns {', '.join(data[numeric_cols].columns[data[numeric_cols].isna().all()])} contain only missing values.")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=numeric_cols, index=data.index)

@st.cache_resource
def optimal_k(df: pd.DataFrame, numeric_cols: List[str], missing_value_strategy: str) -> int:
    """Suggests optimal number of clusters."""
    try:
        data = preprocess_clustering(df, numeric_cols, missing_value_strategy)
        if len(data) < 2:
            raise ValueError("Insufficient data for silhouette score calculation.")
        silhouettes = []
        for k in range(2, min(11, len(data))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            silhouettes.append(silhouette_score(data, labels))
        return silhouettes.index(max(silhouettes)) + 2 if silhouettes else 3
    except ValueError as ve:
        logging.error(f"Optimal k error: {ve}")
        st.session_state.error_messages.append(f"Failed to determine optimal k: {str(ve)}. Using default k=3.")
        return 3

@st.cache_resource
def perform_kmeans(df: pd.DataFrame, numeric_cols: List[str], k: int, missing_value_strategy: str) -> Dict[str, Any]:
    """Performs K-Means clustering."""
    try:
        if len(numeric_cols) < 2:
            raise ValueError("At least 2 numeric columns required for clustering.")
        data = preprocess_clustering(df, numeric_cols, missing_value_strategy)
        if len(data) < k:
            raise ValueError(f"Insufficient data: only {len(data)} rows, need at least {k}.")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        silhouette = silhouette_score(data, labels) if len(data) > k else -1
        return {
            "labels": labels,
            "centers": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_,
            "silhouette": silhouette,
            "index": data.index
        }
    except ValueError as ve:
        logging.error(f"K-Means error: {ve}")
        st.session_state.error_messages.append(f"K-Means failed: {str(ve)} Select valid numeric columns or clean the data.")
        return {"error": str(ve)}
    except Exception as e:
        logging.error(f"K-Means unexpected error: {e}")
        st.session_state.error_messages.append(f"Unexpected error in K-Means: {str(e)}")
        return {"error": str(e)}

@st.cache_resource
def perform_dbscan(df: pd.DataFrame, numeric_cols: List[str], eps: float, min_samples: int, missing_value_strategy: str) -> Dict[str, Any]:
    """Performs DBSCAN clustering."""
    try:
        if len(numeric_cols) < 2:
            raise ValueError("At least 2 numeric columns required for clustering.")
        data = preprocess_clustering(df, numeric_cols, missing_value_strategy)
        if len(data) < min_samples:
            raise ValueError(f"Insufficient data: only {len(data)} rows, need at least {min_samples}.")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette = silhouette_score(data, labels) if n_clusters > 1 and len(data) > n_clusters else -1
        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "silhouette": silhouette,
            "index": data.index
        }
    except ValueError as ve:
        logging.error(f"DBSCAN error: {ve}")
        st.session_state.error_messages.append(f"DBSCAN failed: {str(ve)} Adjust eps/min_samples or clean the data.")
        return {"error": str(ve)}
    except Exception as e:
        logging.error(f"DBSCAN unexpected error: {e}")
        st.session_state.error_messages.append(f"Unexpected error in DBSCAN: {str(e)}")
        return {"error": str(e)}

def preprocess_ml(df: pd.DataFrame, x_cols: List[str], y_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocesses data for supervised ML."""
    cols_to_use = x_cols + [y_col]
    df_clean = df[cols_to_use].dropna()
    if df_clean.empty:
        raise ValueError(f"No valid data after dropping rows with missing values in {cols_to_use}.")
    X = df_clean[x_cols].copy()
    y = df_clean[y_col]
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if categorical_cols.size > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), index=X.index,
                                 columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(categorical_cols, axis=1).join(X_encoded)
    numeric_cols_x = X.select_dtypes(include=np.number).columns
    if len(numeric_cols_x) > 0:
        scaler = StandardScaler()
        X[numeric_cols_x] = scaler.fit_transform(X[numeric_cols_x])
    return X, y

@st.cache_resource
def perform_random_forest(df: pd.DataFrame, x_cols: List[str], y_col: str, model_type: str, tune: bool) -> Dict[str, Any]:
    """Performs Random Forest."""
    try:
        if model_type == "regression" and not pd.api.types.is_numeric_dtype(df[y_col].dtype):
            raise ValueError("Target must be numeric for regression.")
        if model_type == "classification" and pd.api.types.is_numeric_dtype(df[y_col].dtype) and len(df[y_col].unique()) > 10:
            raise ValueError("Target has too many unique values for classification.")
        X, y = preprocess_ml(df, x_cols, y_col)
        if len(X) < 20:
            raise ValueError(f"Insufficient data: only {len(X)} rows available.")
        if model_type == "classification":
            y, _ = pd.factorize(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42, n_jobs=-1) if model_type == "classification" else RandomForestRegressor(random_state=42, n_jobs=-1)
        if model_type == "classification":
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv_strategy = 3
        if tune:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=cv_strategy, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_params = None
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {}
        if model_type == "regression":
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
            metrics["R2"] = r2_score(y_test, y_pred)
            cv_score = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2', n_jobs=-1).mean()
        else:
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            metrics["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics["F1"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cv_score = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1).mean()
        metrics['CV_Score'] = cv_score
        return {
            "model": model,
            "metrics": metrics,
            "feature_importances": model.feature_importances_,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "best_params": best_params
        }
    except ValueError as ve:
        logging.error(f"Random Forest error: {ve}")
        st.session_state.error_messages.append(f"Random Forest failed: {str(ve)}")
        return {"error": str(ve)}
    except Exception as e:
        logging.error(f"Random Forest unexpected error: {e}")
        st.session_state.error_messages.append(f"Unexpected error in Random Forest: {str(e)}")
        return {"error": str(e)}

@st.cache_resource
def perform_xgboost(df: pd.DataFrame, x_cols: List[str], y_col: str, model_type: str, tune: bool) -> Dict[str, Any]:
    """Performs XGBoost."""
    try:
        if model_type == "regression" and not pd.api.types.is_numeric_dtype(df[y_col].dtype):
            raise ValueError("Target must be numeric for regression.")
        if model_type == "classification" and pd.api.types.is_numeric_dtype(df[y_col].dtype) and len(df[y_col].unique()) > 10:
            raise ValueError("Target has too many unique values for classification.")
        X, y = preprocess_ml(df, x_cols, y_col)
        if len(X) < 20:
            raise ValueError(f"Insufficient data: only {len(X)} rows available.")
        if model_type == "classification":
            y, _ = pd.factorize(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(random_state=42, n_jobs=-1) if model_type == "classification" else xgb.XGBRegressor(random_state=42, n_jobs=-1)
        if model_type == "classification":
            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv_strategy = 3
        if tune:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.7, 0.8, 1.0]
            }
            search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=cv_strategy, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_params = None
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {}
        if model_type == "regression":
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
            metrics["R2"] = r2_score(y_test, y_pred)
            cv_score = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2', n_jobs=-1).mean()
        else:
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            metrics["Precision"] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics["Recall"] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics["F1"] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cv_score = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1).mean()
        metrics['CV_Score'] = cv_score
        return {
            "model": model,
            "metrics": metrics,
            "feature_importances": model.feature_importances_,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "best_params": best_params
        }
    except ValueError as ve:
        logging.error(f"XGBoost error: {ve}")
        st.session_state.error_messages.append(f"XGBoost failed: {str(ve)}")
        return {"error": str(ve)}
    except Exception as e:
        logging.error(f"XGBoost unexpected error: {e}")
        st.session_state.error_messages.append(f"Unexpected error in XGBoost: {str(e)}")
        return {"error": str(e)}

def generate_dashboard_html(df: pd.DataFrame, eda_data: Dict[str, Any], model_results: Dict[str, Any], figs: List[Any]) -> str:
    """Generates HTML for dashboard export."""
    html = "<html><body><h1>Dashboard Export</h1>"
    html += f"<h2>EDA Summary</h2><p>Shape: {eda_data['shape']}</p>"
    html += "<h3>Insights</h3><ul>"
    for insight in eda_data['insights']:
        html += f"<li>{insight}</li>"
    html += "</ul>"
    for algo, result in model_results.items():
        if result and "error" not in result:
            html += f"<h2>{algo}</h2><p>Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in result['metrics'].items()])}</p>"
            if result.get("best_params"):
                html += f"<p>Best Parameters: {result['best_params']}</p>"
    html += "<h2>Charts</h2>"
    for fig in figs:
        if fig is not None:
            if isinstance(fig, plt.Figure):
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                html += f'<img src="data:image/png;base64,{img_base64}" />'
            else:
                img_bytes = fig.to_image(format="png")
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                html += f'<img src="data:image/png;base64,{img_base64}" />'
    html += "</body></html>"
    return html

# --- Main Application ---

def main() -> None:
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="Data Visualization Platform", page_icon="⚡️", layout="wide", initial_sidebar_state="expanded")

    # --- Custom CSS ---
    st.markdown("""
        <style>
        .stApp { font-family: Arial, sans-serif; }
        @media (max-width: 600px) {
            .stSidebar { width: 100% !important; }
            .stButton > button { width: 100%; }
        }
        .error-box { background-color: #ffcccc; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚡️ Data Visualization Platform")

    # --- Display and Clear Errors ---
    if st.session_state.error_messages:
        for msg in st.session_state.error_messages:
            st.markdown(f'<div class="error-box">{msg}</div>', unsafe_allow_html=True)
        if st.button("Clear Errors"):
            st.session_state.error_messages = []
            st.rerun()

    # --- Data Upload ---
    uploaded_files = st.file_uploader("Upload your files", type=['csv', 'xlsx', 'json'], accept_multiple_files=True)

    if not uploaded_files:
        st.info("Please upload files to begin.")
        return

    # Reset state if new files uploaded
    current_file_names = [f.name for f in uploaded_files]
    if st.session_state.last_uploaded_files != current_file_names:
        st.session_state.chart_configs = []
        st.session_state.filter_state = {}
        st.session_state.last_uploaded_files = current_file_names
        st.session_state.dfs = {}
        st.session_state.error_messages = []
        st.session_state.data_loaded = False

    with status_message("Loading files..."):
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext == 'csv':
                file_type = "CSV"
            elif file_ext in ['xls', 'xlsx']:
                file_type = "Excel"
            elif file_ext == 'json':
                file_type = "JSON"
            else:
                st.session_state.error_messages.append(f"Unsupported file extension for {uploaded_file.name}")
                continue
            df = load_data(uploaded_file, file_type)
            if df is not None:
                st.session_state.dfs[uploaded_file.name] = df
                st.session_state.data_loaded = True

    if not st.session_state.dfs:
        st.session_state.error_messages.append("No valid files loaded. Please check file formats.")
        st.rerun()

    # Select dataset
    selected_file = st.selectbox("Select Dataset", list(st.session_state.dfs.keys()))
    df = st.session_state.dfs[selected_file]
    st.session_state.selected_df = df

    # Column detection
    numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if not df[col].isna().all()]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    # --- EDA Summary ---
    st.header("Exploratory Data Analysis")
    with status_message("Computing EDA summary..."):
        eda_data = compute_eda_summary(df)
    st.write(f"**Dataset Shape**: {eda_data['shape'][0]} rows × {eda_data['shape'][1]} columns")
    with st.expander("Statistical Summary"):
        st.dataframe(pd.DataFrame(eda_data['describe']))
    with st.expander("Data Types"):
        st.dataframe(pd.Series(eda_data['dtypes'], name='dtype'))
    with st.expander("Missing Values"):
        st.dataframe(pd.Series(eda_data['missing'], name='missing'))
    with st.expander("Insights"):
        for insight in eda_data['insights']:
            st.write(insight)
    st.download_button(
        label="Download EDA Summary as CSV",
        data=pd.DataFrame(eda_data['describe']).to_csv(index=True),
        file_name="eda_summary.csv",
        mime="text/csv"
    )

    # --- Data Cleaning ---
    st.header("Data Cleaning")
    with st.expander("Edit Data"):
        edited_df = st.data_editor(df, num_rows="dynamic", column_config={col: {"editable": True} for col in df.columns}, use_container_width=True)
        if st.button("Apply Changes"):
            with status_message("Applying data changes..."):
                try:
                    df = edited_df.copy()
                    st.session_state.dfs[selected_file] = df
                    st.session_state.error_messages.append("Data changes applied successfully.")
                    st.rerun()
                except Exception as e:
                    st.session_state.error_messages.append(f"Failed to apply changes: {str(e)}")
                    st.rerun()

        missing_action = st.selectbox("Handle Missing Values", ["None", "Drop Rows", "Fill with Mean"])
        if missing_action != "None" and st.button("Apply Missing Value Action"):
            with status_message("Handling missing values..."):
                try:
                    if missing_action == "Drop Rows":
                        df = df.dropna()
                    elif missing_action == "Fill with Mean":
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    st.session_state.dfs[selected_file] = df
                    st.session_state.error_messages.append(f"Applied {missing_action} to missing values.")
                    st.rerun()
                except Exception as e:
                    st.session_state.error_messages.append(f"Failed to handle missing values: {str(e)}")
                    st.rerun()

    # --- Machine Learning ---
    st.header("Machine Learning")
    model_results = {}
    with st.expander("Run Machine Learning"):
        algorithm = st.selectbox("Select Algorithm", ALGORITHM_OPTIONS)
        selected_cols = st.multiselect(
            "Select Features",
            numeric_cols,
            default=numeric_cols if len(numeric_cols) >= 2 else numeric_cols,
            help="Select at least 2 numeric columns for clustering or input features for supervised models."
        )

        if algorithm in ["K-Means Clustering", "DBSCAN Clustering"]:
            missing_value_strategy = st.selectbox("Handle Missing Values", MISSING_VALUE_OPTIONS, index=0)
            if algorithm == "K-Means Clustering":
                auto_k = st.checkbox("Auto-Suggest k")
                k = optimal_k(df, selected_cols, missing_value_strategy) if auto_k else st.slider("Number of Clusters (k)", 2, 10, 3)
                if auto_k:
                    st.info(f"Suggested k: {k}")
            else:
                eps = st.slider("Epsilon (eps)", 0.1, 3.0, 0.5, help="Distance threshold for DBSCAN clustering")
                min_samples = st.slider("Minimum Samples", 2, 20, 5, help="Minimum points to form a cluster")
        else:
            model_type = st.selectbox("Model Type", ["classification", "regression"])
            x_cols = st.multiselect("Input Features (X)", numeric_cols + categorical_cols, default=selected_cols)
            y_col = st.selectbox("Target (Y)", df.columns)
            tune = st.checkbox("Tune Hyperparameters")
            if y_col:
                unique_y = len(df[y_col].dropna().unique())
                suggested_type = "classification" if not pd.api.types.is_numeric_dtype(df[y_col]) or unique_y <= 10 else "regression"
                st.info(f"Suggested model type: {suggested_type} (based on target '{y_col}' with {unique_y} unique values)")

        if st.button("Run Model"):
            with status_message(f"Running {algorithm}..."):
                if algorithm in ["K-Means Clustering", "DBSCAN Clustering"] and len(selected_cols) < 2:
                    st.session_state.error_messages.append("At least 2 numeric columns required for clustering.")
                elif algorithm in ["Random Forest", "XGBoost"] and (not x_cols or not y_col or y_col in x_cols):
                    st.session_state.error_messages.append("Select valid input features (X) and target (Y). Target cannot be an input feature.")
                else:
                    df_for_model = df[selected_cols if algorithm in ["K-Means Clustering", "DBSCAN Clustering"] else x_cols + [y_col]].copy()
                    if algorithm in ["K-Means Clustering", "DBSCAN Clustering"]:
                        if missing_value_strategy == "Drop Rows":
                            df_for_model = df_for_model.dropna()
                        else:
                            df_for_model[selected_cols] = df_for_model[selected_cols].fillna(df_for_model[selected_cols].mean())
                        if df_for_model[selected_cols].isna().all().any():
                            st.session_state.error_messages.append(f"Columns {', '.join(df_for_model[selected_cols].columns[df_for_model[selected_cols].isna().all()])} contain only missing values.")
                        elif algorithm == "K-Means Clustering" and len(df_for_model) < k:
                            st.session_state.error_messages.append(f"Insufficient data: only {len(df_for_model)} rows, need at least {k}.")
                        elif algorithm == "DBSCAN Clustering" and len(df_for_model) < min_samples:
                            st.session_state.error_messages.append(f"Insufficient data: only {len(df_for_model)} rows, need at least {min_samples}.")
                        else:
                            if algorithm == "K-Means Clustering":
                                result = perform_kmeans(df, selected_cols, k, missing_value_strategy)
                                if "error" not in result:
                                    df['cluster'] = np.nan
                                    df.loc[result["index"], 'cluster'] = result["labels"]
                                    st.session_state.dfs[selected_file] = df
                                    st.success(f"K-Means complete! Inertia: {result['inertia']:.2f}, Silhouette: {result['silhouette']:.4f}")
                                    fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1], color='cluster', title="K-Means Clusters")
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.write("Cluster Centers:")
                                    st.dataframe(pd.DataFrame(result["centers"], columns=selected_cols))
                                    model_results["K-Means Clustering"] = result
                            else:
                                result = perform_dbscan(df, selected_cols, eps, min_samples, missing_value_strategy)
                                if "error" not in result:
                                    df['cluster'] = np.nan
                                    df.loc[result["index"], 'cluster'] = result["labels"]
                                    st.session_state.dfs[selected_file] = df
                                    st.success(f"DBSCAN complete! Clusters: {result['n_clusters']}, Silhouette: {result['silhouette']:.4f}")
                                    fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1], color='cluster', title="DBSCAN Clusters")
                                    st.plotly_chart(fig, use_container_width=True)
                                    model_results["DBSCAN Clustering"] = result
                    else:
                        if len(df_for_model.dropna()) < 20:
                            st.session_state.error_messages.append(f"Insufficient data: only {len(df_for_model.dropna())} rows after dropping missing values.")
                        else:
                            result = (perform_random_forest if algorithm == "Random Forest" else perform_xgboost)(df, x_cols, y_col, model_type, tune)
                            if "error" not in result:
                                st.success(f"{algorithm} complete! Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in result['metrics'].items()])}")
                                if result.get("best_params"):
                                    st.write(f"Best Parameters: {result['best_params']}")
                                feat_imp = pd.DataFrame({'Feature': result['X_test'].columns, 'Importance': result['feature_importances']})
                                fig = px.bar(feat_imp, x='Feature', y='Importance', title="Feature Importances")
                                st.plotly_chart(fig, use_container_width=True)
                                fig_pred = px.scatter(x=result['y_test'], y=result['y_pred'], title="Actual vs Predicted")
                                fig_pred.add_scatter(x=result['y_test'], y=result['y_test'], mode='lines', name='Ideal')
                                st.plotly_chart(fig_pred, use_container_width=True)
                                model_results[algorithm] = result

    # --- Sidebar: Chart Configuration ---
    with st.sidebar:
        st.header("⚙️ Controls")
        st.subheader("Chart Configuration")
        suggested_charts = suggest_chart_type(df, numeric_cols, categorical_cols)
        st.info(f"Suggested charts: {', '.join(suggested_charts)}")

        if st.button("Add New Chart"):
            st.session_state.chart_configs.append({"chart_type": CHART_OPTIONS[0], "id": len(st.session_state.chart_configs)})

        with st.expander("Chart Styling"):
            theme = st.selectbox("Theme", THEME_OPTIONS, key="theme")
            color_palette = st.selectbox("Color Palette", COLOR_PALETTES, key="color")
            custom_title = st.text_input("Chart Title", "My Visualization")

    # --- Generated Visualizations ---
    st.header("Generated Visualizations")
    figs = []
    if st.session_state.chart_configs:
        tabs = st.tabs([f"Chart {i + 1}" for i in range(len(st.session_state.chart_configs))])
        for idx, tab in enumerate(tabs):
            with tab:
                config = st.session_state.chart_configs[idx]
                chart_type = st.radio("Select Chart Type", CHART_OPTIONS, index=CHART_OPTIONS.index(config["chart_type"]), key=f"chart_type_{idx}")
                st.session_state.chart_configs[idx]["chart_type"] = chart_type

                x_label = st.text_input("X-axis Label", "", key=f"x_label_{idx}")
                y_label = st.text_input("Y-axis Label", "", key=f"y_label_{idx}")
                font_size = st.slider("Font Size", 8, 20, 12, key=f"font_size_{idx}")
                legend_position = st.selectbox("Legend Position", LEGEND_POSITIONS, key=f"legend_pos_{idx}")

                x_axis, y_axis, agg_func = None, None, None
                if chart_type in ["Scatter Plot", "Line Chart"]:
                    x_axis = st.selectbox("Select X-axis", numeric_cols, key=f"x_axis_{idx}")
                    y_axis = st.selectbox("Select Y-axis", numeric_cols, key=f"y_axis_{idx}")
                elif chart_type == "Bar Chart":
                    x_axis = st.selectbox("Group by (X-axis)", categorical_cols, key=f"x_axis_{idx}")
                    y_axis = st.selectbox("Aggregate column (Y-axis)", numeric_cols, key=f"y_axis_{idx}")
                    agg_func = st.selectbox("Aggregation function", ["mean", "sum", "count"], key=f"agg_func_{idx}")
                elif chart_type == "Histogram":
                    x_axis = st.selectbox("Select column", numeric_cols, key=f"x_axis_{idx}")
                elif chart_type == "Box Plot":
                    y_axis = st.selectbox("Select column (Y-axis)", numeric_cols, key=f"y_axis_{idx}")
                    x_axis = st.selectbox("Optional: Group by (X-axis)", [None] + categorical_cols, key=f"x_axis_{idx}")
                elif chart_type == "Pie Chart":
                    x_axis = st.selectbox("Category", categorical_cols, key=f"x_axis_{idx}")
                    y_axis = st.selectbox("Values", numeric_cols, key=f"y_axis_{idx}")
                    agg_func = 'sum'

                df_to_plot = df
                fig = None
                try:
                    if chart_type in ["Bar Chart", "Pie Chart"] and x_axis and y_axis and agg_func:
                        df_to_plot = df.groupby(x_axis)[y_axis].agg(agg_func).reset_index()
                        if df_to_plot.empty:
                            st.session_state.error_messages.append("Aggregation resulted in an empty dataset.")
                            continue

                    if chart_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
                        corr_matrix = df[numeric_cols].corr()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(corr_matrix, annot=True, cmap=color_palette.lower(), fmt=".2f", ax=ax)
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.tick_params(labelsize=font_size)
                        st.pyplot(fig)
                    else:
                        is_ready_to_plot = (x_axis and y_axis) or (chart_type == "Histogram" and x_axis) or (chart_type == "Box Plot" and y_axis)
                        if not is_ready_to_plot:
                            st.session_state.error_messages.append("Please configure the required chart settings.")
                            continue

                        plot_args = {
                            'data_frame': df_to_plot,
                            'template': theme,
                            'color_discrete_sequence': px.colors.sequential.__getattribute__(color_palette),
                            'labels': {'x': x_label, 'y': y_label}
                        }
                        plot_functions = {
                            "Scatter Plot": px.scatter,
                            "Line Chart": px.line,
                            "Bar Chart": px.bar,
                            "Histogram": px.histogram,
                            "Box Plot": px.box,
                            "Pie Chart": px.pie
                        }
                        if chart_type in plot_functions:
                            plot_func = plot_functions[chart_type]
                            if chart_type == "Histogram":
                                fig = plot_func(**plot_args, x=x_axis, title=custom_title)
                            elif chart_type == "Pie Chart":
                                fig = plot_func(**plot_args, names=x_axis, values=y_axis, title=custom_title)
                            elif chart_type == "Box Plot":
                                fig = plot_func(**plot_args, x=x_axis, y=y_axis, title=custom_title)
                            else:
                                fig = plot_func(**plot_args, x=x_axis, y=y_axis, title=custom_title)
                            fig.update_layout(font=dict(size=font_size), showlegend=legend_position != "none")
                            st.plotly_chart(fig, use_container_width=True)
                    if fig:
                        figs.append(fig)
                except Exception as e:
                    st.session_state.error_messages.append(f"Chart rendering failed: {str(e)}")

                st.subheader("Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = convert_df_to_csv(df_to_plot)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv_data,
                        file_name=f"{chart_type.lower().replace(' ', '_')}_data_{idx}.csv",
                        mime="text/csv"
                    )
                with col2:
                    if isinstance(fig, plt.Figure):
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        img_bytes = buf.getvalue()
                        st.download_button(
                            label="Download Chart as PNG",
                            data=img_bytes,
                            file_name=f"{chart_type.lower().replace(' ', '_')}_chart_{idx}.png",
                            mime="image/png"
                        )
                    elif fig:
                        chart_json = fig.to_json()
                        st.download_button(
                            label="Download Chart as JSON",
                            data=chart_json,
                            file_name=f"{chart_type.lower().replace(' ', '_')}_chart_{idx}.json",
                            mime="application/json"
                        )
    else:
        st.info("Add a new chart using the sidebar to visualize your data.")

    # --- Data Preview ---
    with st.expander("Data Preview"):
        st.write(f"**Data**: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))
        st.write(f"**Numeric Columns**: {', '.join(numeric_cols)}")
        st.write(f"**Categorical Columns**: {', '.join(categorical_cols)}")
        if datetime_cols:
            st.write(f"**Datetime Columns**: {', '.join(datetime_cols)}")

    # --- Dashboard Export ---
    st.header("Dashboard Export")
    if st.button("Export Dashboard to HTML"):
        with status_message("Generating dashboard export..."):
            html_content = generate_dashboard_html(df, eda_data, model_results, figs)
            st.download_button(
                label="Download Dashboard HTML",
                data=html_content,
                file_name="dashboard_export.html",
                mime="text/html"
            )

    # --- Help ---
    with st.sidebar.expander("Help & Tutorials"):
        st.markdown("""
            **Welcome to the Data Visualization Platform!**
            - Upload CSV, Excel, or JSON files (ensure proper formatting, e.g., correct CSV separator).
            - View EDA, edit data, and run ML models (K-Means, DBSCAN, Random Forest, XGBoost).
            - For clustering, select at least 2 numeric columns and use 'Fill with Mean' to avoid data loss.
            - For supervised models, choose valid input features and a target (classification for categorical, regression for numeric).
            - Export charts and dashboard as CSV, PNG, or HTML.
            - **Troubleshooting**:
              - **File loading errors**: Ensure correct separator (e.g., ',' or ';') for CSV.
              - **Clustering failures**: Select valid numeric columns, use 'Fill with Mean' if 'Drop Rows' removes data.
              - **Model failures**: Ensure target is appropriate (numeric for regression, categorical for classification).
              - **Errors persist**: Click 'Clear Errors' and verify selections.
        """)

if __name__ == "__main__":
    main()