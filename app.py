import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
import io
import json
import openpyxl
import duckdb
import numpy as np
from datetime import datetime
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score, \
    silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE
import base64

# --- Constants for Magic Strings ---
FILE_TYPES: List[str] = ["CSV", "Excel", "JSON"]
CHART_OPTIONS: List[str] = [
    "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot",
    "Correlation Heatmap", "Pie Chart", "Violin Plot", "Map View"
]
AGG_OPTIONS: List[str] = ['mean', 'sum', 'median', 'count', 'min', 'max']
THEME_OPTIONS: List[str] = ["plotly", "plotly_dark", "seaborn"]
COLOR_PALETTES: List[str] = ["Viridis", "Plasma", "Inferno", "Magma"]
LEGEND_POSITIONS: List[str] = ["top", "bottom", "left", "right", "none"]

# --- Setup Logging for Error Tracking ---
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Session State for Persistence ---
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


# --- Helper Functions ---

@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile], file_type: str) -> Optional[
    pd.DataFrame]:
    """Loads data from various file formats."""
    try:
        if file_type == "CSV":
            return pd.read_csv(uploaded_file)
        elif file_type == "Excel":
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_type == "JSON":
            return pd.read_json(uploaded_file)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        logging.error(f"Data loading error: {e}")
        st.error(f"Failed to load file: {e}")
        return None


@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Converts DataFrame to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')


def find_lat_lon_columns(columns: pd.Index) -> Tuple[Optional[str], Optional[str]]:
    """Identifies latitude/longitude columns."""
    lat_col, lon_col = None, None
    lat_pattern = re.compile(r'^(lat|latitude)$', re.IGNORECASE)
    lon_pattern = re.compile(r'^(lon|lng|long|longitude)$', re.IGNORECASE)
    for col in columns:
        if not lat_col and lat_pattern.match(col): lat_col = col
        if not lon_col and lon_pattern.match(col): lon_col = col
    return lat_col, lon_col


def get_widget_value(param_name: str, default_value: Any, param_type: type = str) -> Any:
    """Gets widget value from URL params with type casting."""
    param_value = st.query_params.get(param_name)
    if param_value is None:
        return default_value
    try:
        if param_type in (list, tuple):
            return json.loads(param_value)
        return param_type(param_value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default_value


def suggest_chart_type(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """AI-driven chart type suggestions based on data."""
    if len(numeric_cols) >= 2:
        return ["Correlation Heatmap", "Scatter Plot", "Box Plot"]
    elif categorical_cols and numeric_cols:
        return ["Bar Chart", "Box Plot"]
    elif numeric_cols:
        return ["Histogram", "Box Plot"]
    return ["Table View"]


@st.cache_data
def compute_eda_summary(df: pd.DataFrame, sample_size: int = 50000) -> Dict[str, Any]:
    """Generates a simple EDA summary using pandas."""
    if df.empty:
        return {
            'shape': (0, 0),
            'describe': {},
            'dtypes': {},
            'missing': {},
            'insights': {}
        }
    sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
    numeric_cols = sampled_df.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = sampled_df[numeric_cols].corr()
    high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
    insights = []
    for col1 in high_corr.columns:
        for col2 in high_corr.index:
            if high_corr.loc[col2, col1] and col1 != col2:
                insights.append(
                    f"High correlation between {col1} and {col2} ({corr_matrix.loc[col2, col1]:.2f}). Consider investigating multicollinearity.")
    if not insights:
        insights.append(
            "No high correlations found in the sample. For large datasets, consider increasing sample size or reviewing full correlations if memory allows.")
    return {
        'shape': df.shape,
        'describe': sampled_df.describe(include='all').to_dict(),
        'dtypes': sampled_df.dtypes.astype(str).to_dict(),
        'missing': sampled_df.isnull().sum().to_dict(),
        'insights': insights
    }


def apply_duckdb_filters(df: pd.DataFrame, filter_state: Dict[str, Any], categorical_cols: List[str],
                         numeric_cols: List[str], datetime_cols: List[str],
                         regex_conditions: Dict[str, str]) -> pd.DataFrame:
    """Applies filters using DuckDB for performance."""
    con = duckdb.connect()
    con.register('df_filtered', df)
    query_parts = ["SELECT * FROM df_filtered"]
    conditions = []

    for col in categorical_cols:
        if f"filter_{col}" in filter_state and filter_state[f"filter_{col}"]:
            # --- FIX IS HERE ---
            # Escape single quotes in each value by replacing ' with ''
            vals = ", ".join(f"'{str(v).replace("'", "''")}'" for v in filter_state[f"filter_{col}"])
            conditions.append(f'"{col}" IN ({vals})')

    # ... (the rest of the function remains the same) ...
    for col in numeric_cols:
        if f"filter_{col}" in filter_state:
            min_val, max_val = filter_state[f"filter_{col}"]
            conditions.append(f'"{col}" BETWEEN {min_val} AND {max_val}')
    for col in datetime_cols:
        if f"filter_{col}" in filter_state:
            start_date, end_date = filter_state[f"filter_{col}"]
            conditions.append(f'"{col}" BETWEEN \'{start_date}\' AND \'{end_date}\'')
    for col, pattern in regex_conditions.items():
        conditions.append(f'REGEXP_MATCHES("{col}", \'{pattern}\')')

    if conditions:
        query_parts.append("WHERE " + " AND ".join(conditions))

    query = " ".join(query_parts)
    return con.execute(query).fetchdf()


def preprocess_kmeans(data: pd.DataFrame, numeric_cols: List[str], remove_outliers: bool = False) -> pd.DataFrame:
    """Preprocesses data for K-Means with scaling and optional outlier removal."""
    data = data[numeric_cols].dropna()
    if remove_outliers:
        iso = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso.fit_predict(data)
        data = data[outliers == 1]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=numeric_cols, index=data.index)


@st.cache_resource
def optimal_k(df: pd.DataFrame, numeric_cols: List[str], max_k: int = 10) -> int:
    """Suggests optimal number of clusters using silhouette score."""
    data = preprocess_kmeans(df, numeric_cols)
    silhouettes = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouettes.append(silhouette_score(data, labels))
    return silhouettes.index(max(silhouettes)) + 2


@st.cache_resource
def perform_kmeans(df: pd.DataFrame, numeric_cols: List[str], k: int = 3, preprocess: bool = False,
                   remove_outliers: bool = False) -> Dict[str, Any]:
    """Performs K-Means clustering with optional preprocessing."""
    if len(numeric_cols) < 2:
        return {"error": "At least 2 numeric columns required for clustering."}
    data = preprocess_kmeans(df, numeric_cols, remove_outliers) if preprocess else df[numeric_cols].dropna()
    if len(data) < k:
        return {"error": "Insufficient data for clustering."}
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    return {"labels": labels, "centers": kmeans.cluster_centers_, "inertia": kmeans.inertia_, "index": data.index}


def preprocess_rf(df: pd.DataFrame, x_cols: List[str], y_col: str, model_type: str, remove_outliers: bool = False) -> \
        Tuple[pd.DataFrame, pd.Series]:
    """Preprocesses data for Random Forest with encoding, scaling, and feature selection."""
    X = df[x_cols].copy()
    y = df[y_col].dropna()
    if remove_outliers:
        iso = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso.fit_predict(X)
        X = X[outliers == 1]
        y = y[outliers == 1]
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if categorical_cols.size > 0 and model_type == "classification":
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), index=X.index,
                                 columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(categorical_cols, axis=1).join(X_encoded)
    if model_type == "regression":
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    model = RandomForestRegressor(random_state=42) if model_type == "regression" else RandomForestClassifier(
        random_state=42)
    rfe = RFE(model, n_features_to_select=min(10, len(X.columns)))
    X_selected = pd.DataFrame(rfe.fit_transform(X, y), index=X.index)
    return X_selected, y


@st.cache_resource
def tune_random_forest(X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, Any]:
    """Tunes Random Forest hyperparameters using RandomizedSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    scoring = 'r2' if model_type == "regression" else 'f1_weighted'
    model = RandomForestRegressor(random_state=42) if model_type == "regression" else RandomForestClassifier(
        random_state=42)
    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42, scoring=scoring)
    search.fit(X, y)
    best_index = search.best_index_
    cv_split_scores = [search.cv_results_[f'split{i}_test_score'][best_index] for i in range(5)]
    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_mean": search.best_score_,
        "cv_std": search.cv_results_['std_test_score'][best_index],
        "cv_min": min(cv_split_scores)
    }


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, model_type: str) -> Dict[str, float]:
    """Evaluates model with extended metrics."""
    metrics = {}
    if model_type == "regression":
        metrics["MSE"] = mean_squared_error(y_true, y_pred)
        metrics["R2"] = r2_score(y_true, y_pred)
    else:
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["F1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return metrics


@st.cache_resource
def perform_random_forest(df: pd.DataFrame, x_cols: List[str], y_col: str, model_type: str, preprocess: bool = False,
                          tune: bool = False, remove_outliers: bool = False) -> Dict[str, Any]:
    """Performs Random Forest with optional preprocessing and tuning."""
    X, y = preprocess_rf(df, x_cols, y_col, model_type, remove_outliers) if preprocess else (df[x_cols].dropna(),
                                                                                             df[y_col].dropna())
    if len(X) != len(y) or len(X) < 10:
        return {"error": "Insufficient or mismatched data for model training."}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scoring = 'r2' if model_type == "regression" else 'f1_weighted'
    if tune:
        tuning_result = tune_random_forest(X_train, y_train, model_type)
        model = tuning_result["best_model"]
        best_params = tuning_result["best_params"]
        cv_mean = tuning_result["cv_mean"]
        cv_std = tuning_result["cv_std"]
        cv_min = tuning_result["cv_min"]
    else:
        model = RandomForestRegressor(random_state=42) if model_type == "regression" else RandomForestClassifier(
            random_state=42)
        best_params = None
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_min = np.min(cv_scores)
    if not tune:
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_type)
    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": model.feature_importances_ if not preprocess else None,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "best_params": best_params,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_min": cv_min
    }


def time_operation(func: callable, *args, **kwargs) -> Tuple[Any, float]:
    """Times the execution of a function."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def generate_dashboard_html(df: pd.DataFrame, eda_data: Dict[str, Any], cluster_result: Optional[Dict[str, Any]],
                            rf_result: Optional[Dict[str, Any]], figs: List[Any]) -> str:
    """Generates HTML for dashboard export."""
    html = "<html><body><h1>Dashboard Export</h1>"
    html += f"<h2>EDA Summary</h2><p>Shape: {eda_data['shape']}</p>"
    html += "<h3>Insights</h3><ul>"
    for insight in eda_data['insights']:
        html += f"<li>{insight}</li>"
    html += "</ul>"
    if cluster_result and "error" not in cluster_result:
        html += f"<h2>K-Means Clustering</h2><p>Inertia: {cluster_result['inertia']:.2f}</p>"
    if rf_result and "error" not in rf_result:
        html += f"<h2>Random Forest</h2><p>Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in rf_result['metrics'].items()])}</p>"
        if rf_result.get("best_params"):
            html += f"<p>Best Parameters: {rf_result['best_params']}</p>"
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
    """Main function for the enhanced Streamlit app."""
    st.set_page_config(
        page_title="Advanced Data Visualization Platform",
        page_icon="⚡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Apply Custom CSS for Responsiveness and Themes ---
    st.markdown("""
        <style>
        .stApp { font-family: Arial, sans-serif; }
        @media (max-width: 600px) {
            .stSidebar { width: 100% !important; }
            .stButton > button { width: 100%; }
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("⚡️ Advanced Data Visualization Platform")

    # --- Batch Data Upload in Main Section ---
    file_type = st.selectbox("File Type", FILE_TYPES, help="Select file format")
    uploaded_files = st.file_uploader("Upload your files (batch supported)", type=['csv', 'xlsx', 'json'],
                                      accept_multiple_files=True)

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

    with st.spinner("Processing files..."):
        for uploaded_file in uploaded_files:
            df = load_data(uploaded_file, file_type)
            if df is not None:
                st.session_state.dfs[uploaded_file.name] = df

    if not st.session_state.dfs:
        return

    # Select dataset for analysis
    selected_file = st.selectbox("Select Dataset", list(st.session_state.dfs.keys()))
    df = st.session_state.dfs[selected_file]
    st.session_state.selected_df = df

    st.session_state.data_loaded = True
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    lat_col, lon_col = find_lat_lon_columns(df.columns)

    # --- EDA Summary in Main Section ---
    st.header("Exploratory Data Analysis (EDA)")
    eda_data = compute_eda_summary(df)
    st.write(f"**Dataset Shape**: {eda_data['shape'][0]} rows × {eda_data['shape'][1]} columns")
    with st.expander("Statistical Summary (Sampled)"):
        st.dataframe(pd.DataFrame(eda_data['describe']))
    with st.expander("Data Types (Sampled)"):
        st.dataframe(pd.Series(eda_data['dtypes'], name='dtype'))
    with st.expander("Missing Values (Sampled)"):
        st.dataframe(pd.Series(eda_data['missing'], name='missing'))
    with st.expander("Profiling Insights"):
        for insight in eda_data['insights']:
            st.warning(insight)
    csv_data = pd.DataFrame(eda_data['describe']).to_csv(index=True)
    st.download_button(
        label="Download EDA Summary as CSV",
        data=csv_data,
        file_name="eda_summary.csv",
        mime="text/csv"
    )

    # --- Data Cleaning in Main Section ---
    st.header("Data Cleaning")
    with st.expander("Clean Your Data"):
        edited_df = st.data_editor(
            df.reset_index(drop=True),
            num_rows="dynamic",
            column_config={col: {"editable": True} for col in df.columns},
            use_container_width=True
        )
        if st.button("Apply Changes"):
            df = edited_df.copy()
            st.session_state.dfs[selected_file] = df
            st.success("Data changes applied!")

        # Handle missing values
        missing_action = st.selectbox("Handle Missing Values",
                                      ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])
        if missing_action != "None" and st.button("Apply Missing Value Action"):
            if missing_action == "Drop Rows":
                df = df.dropna()
            elif missing_action == "Fill with Mean":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif missing_action == "Fill with Median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            st.session_state.dfs[selected_file] = df
            st.success(f"Applied {missing_action} to missing values.")

    # --- ML Features: K-Means Clustering and Random Forest ---
    st.header("Machine Learning")
    with st.expander("Run K-Means Clustering"):
        auto_k = st.checkbox("Auto-Suggest k")
        remove_outliers_kmeans = st.checkbox("Remove Outliers for Clustering")
        preprocess_kmeans_flag = st.checkbox("Preprocess Features for Clustering")
        k = optimal_k(df, numeric_cols) if auto_k else st.slider("Number of Clusters (k)", 2, 10, 3)
        if auto_k:
            st.info(f"Suggested number of clusters: {k}")
        if st.button("Run Clustering"):
            with st.spinner("Running K-Means..."):
                cluster_result = perform_kmeans(df, numeric_cols, k, preprocess=preprocess_kmeans_flag,
                                                remove_outliers=remove_outliers_kmeans)
            if "error" in cluster_result:
                st.error(cluster_result["error"])
            else:
                df['cluster'] = np.nan
                df.loc[cluster_result["index"], 'cluster'] = cluster_result["labels"]
                st.session_state.dfs[selected_file] = df
                st.success(f"Clustering complete! Inertia: {cluster_result['inertia']:.2f}")
                fig_cluster = px.scatter(df, x=numeric_cols[0] if numeric_cols else None,
                                         y=numeric_cols[1] if len(numeric_cols) > 1 else None, color='cluster',
                                         title="K-Means Clusters")
                st.plotly_chart(fig_cluster)
                st.write("Cluster Centers:")
                st.dataframe(pd.DataFrame(cluster_result["centers"], columns=numeric_cols))

                # Additional clustering metrics
                data = preprocess_kmeans(df, numeric_cols, remove_outliers_kmeans) if preprocess_kmeans_flag else df[numeric_cols].dropna()
                labels = cluster_result["labels"]
                sil_score = silhouette_score(data, labels)
                db_index = davies_bouldin_score(data, labels)
                st.write(f"Silhouette Score: {sil_score:.4f}")
                st.write(f"Davies-Bouldin Index: {db_index:.4f}")

                # Elbow method plot
                max_k_elbow = 10
                inertias = []
                for kk in range(1, max_k_elbow + 1):
                    km = KMeans(n_clusters=kk, random_state=42)
                    km.fit(data)
                    inertias.append(km.inertia_)
                fig_elbow = px.line(x=range(1, max_k_elbow + 1), y=inertias, markers=True, title="Elbow Method")
                st.plotly_chart(fig_elbow)

    with st.expander("Run Random Forest Model"):
        rf_type = st.selectbox("Model Type", ["regression", "classification"])
        rf_x_cols = st.multiselect("Input Features (X)", numeric_cols + categorical_cols)
        rf_y_col = st.selectbox("Target (Y)", df.columns)
        preprocess_rf_flag = st.checkbox("Preprocess Features for Random Forest")
        tune_rf_flag = st.checkbox("Tune Hyperparameters (may take 10-30s)")
        remove_outliers_rf = st.checkbox("Remove Outliers for Random Forest")
        if rf_y_col:
            unique_y = len(df[rf_y_col].unique())
            suggested_type = "regression" if pd.api.types.is_numeric_dtype(
                df[rf_y_col]) else "classification" if unique_y <= 10 else None
            if suggested_type:
                st.info(
                    f"Suggested model type: {suggested_type} (based on target '{rf_y_col}' with {unique_y} unique values)")
        if st.button("Run Random Forest"):
            if not rf_x_cols or not rf_y_col:
                st.error("Please select input features and target.")
            else:
                with st.spinner("Training Random Forest..."):
                    try:
                        rf_result = perform_random_forest(df, rf_x_cols, rf_y_col, rf_type,
                                                          preprocess=preprocess_rf_flag,
                                                          tune=tune_rf_flag, remove_outliers=remove_outliers_rf)

                        if "error" in rf_result:
                            st.error(rf_result["error"])
                        else:
                            st.success(
                                f"Model trained! Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in rf_result['metrics'].items()])}")
                            st.write(
                                f"CV Mean: {rf_result['cv_mean']:.4f}, CV Std: {rf_result['cv_std']:.4f}, CV Min: {rf_result['cv_min']:.4f}")
                            if rf_result.get("best_params"):
                                st.write(f"Best Parameters: {rf_result['best_params']}")
                            if rf_result.get("feature_importances") is not None:
                                feat_imp = pd.DataFrame(
                                    {'Feature': rf_x_cols, 'Importance': rf_result['feature_importances']})
                                fig_imp = px.bar(feat_imp, x='Feature', y='Importance', title="Feature Importances")
                                st.plotly_chart(fig_imp)
                            fig_pred = px.scatter(x=rf_result['y_test'], y=rf_result['y_pred'],
                                                  title="Actual vs Predicted")
                            fig_pred.add_scatter(x=rf_result['y_test'], y=rf_result['y_test'], mode='lines',
                                                 name='Ideal')
                            st.plotly_chart(fig_pred)

                    except ValueError as e:
                        if "could not convert string to float" in str(e):
                            st.error(
                                f"❌ Data Type Error: The model expected numbers but received text. Please check your feature selection. Ensure all input features (X) are numeric or enable 'Preprocess Features' to handle text data automatically. Details: {e}")
                        else:
                            st.error(f"An unexpected value error occurred: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during model training: {e}")
                        logging.error(f"Random Forest Error: {e}")

    # --- Sidebar for Controls: Filtering, Chart Config ---
    with st.sidebar:
        st.header("⚙️ Controls")

        # --- Advanced Filtering ---
        st.subheader("Data Filtering")
        with st.expander("Advanced Filters"):
            for col in categorical_cols:
                unique_vals = list(df[col].unique())
                default_filter = get_widget_value(f"filter_{col}", unique_vals, list)
                selected_vals = st.multiselect(f"Filter by {col}", unique_vals, default=default_filter,
                                               key=f"cat_filter_{col}")
                st.session_state.filter_state[f"filter_{col}"] = selected_vals

            for col in numeric_cols:
                if col not in [lat_col, lon_col]:
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    default_range = get_widget_value(f"filter_{col}", (min_val, max_val), tuple)
                    slider_range = st.slider(f"Filter by {col}", min_val, max_val, default_range,
                                             key=f"num_filter_{col}")
                    st.session_state.filter_state[f"filter_{col}"] = slider_range

            for col in datetime_cols:
                min_date, max_date = df[col].min(), df[col].max()
                # Ensure default dates are python date objects for the widget
                default_dates = (min_date.date(), max_date.date())

                date_range = st.date_input(f"Filter by {col}", value=default_dates, min_value=min_date,
                                           max_value=max_date, key=f"date_filter_{col}")

                # Convert the date objects to full timestamps for comparison
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    st.session_state.filter_state[f"filter_{col}"] = (pd.to_datetime(start_date),
                                                                      pd.to_datetime(end_date))

            # Regex filters
            regex_conditions = {}
            for col in categorical_cols:
                regex_pattern = st.text_input(f"Regex Filter for {col}", "", key=f"regex_{col}")
                if regex_pattern:
                    regex_conditions[col] = regex_pattern

        # --- Chart Configuration ---
        st.subheader("Chart Configuration")
        # Suggest chart types
        suggested_charts = suggest_chart_type(df, numeric_cols, categorical_cols)
        st.info(f"Suggested charts: {', '.join(suggested_charts)}")

        # Multi-chart support
        if st.button("Add New Chart"):
            st.session_state.chart_configs.append(
                {"chart_type": CHART_OPTIONS[0], "id": len(st.session_state.chart_configs)})

        # Chart styling options
        with st.expander("Chart Styling"):
            theme = st.selectbox("Theme", THEME_OPTIONS, key="theme")
            color_palette = st.selectbox("Color Palette", COLOR_PALETTES, key="color")
            custom_title = st.text_input("Chart Title", "My Visualization")

    # --- Apply Filters with DuckDB (timed) ---
    start_duck = time.time()
    df_filtered = apply_duckdb_filters(df, st.session_state.filter_state, categorical_cols, numeric_cols, datetime_cols,
                                       regex_conditions)
    time_duck = time.time() - start_duck

    # pandas comparison for performance metrics
    start_pd = time.time()
    df_pd = df.copy()
    for col in categorical_cols:
        if f"filter_{col}" in st.session_state.filter_state:
            df_pd = df_pd[df_pd[col].isin(st.session_state.filter_state[f"filter_{col}"])]
    for col in numeric_cols:
        if f"filter_{col}" in st.session_state.filter_state:
            min_val, max_val = st.session_state.filter_state[f"filter_{col}"]
            df_pd = df_pd[df_pd[col].between(min_val, max_val)]
    for col in datetime_cols:
        if f"filter_{col}" in st.session_state.filter_state:
            start_date, end_date = st.session_state.filter_state[f"filter_{col}"]
            df_pd = df_pd[df_pd[col].between(start_date, end_date)]
    for col, pattern in regex_conditions.items():
        df_pd = df_pd[df_pd[col].str.contains(pattern, na=False, regex=True)]
    time_pd = time.time() - start_pd

    st.info(f"Filtering Performance: DuckDB ({time_duck:.4f}s) vs Pandas ({time_pd:.4f}s)")

    # --- Display Charts in Main Section ---
    st.header("Generated Visualizations")
    figs = []
    tabs = st.tabs(
        [f"Chart {i + 1}" for i in range(len(st.session_state.chart_configs))]) if st.session_state.chart_configs else [
        st.container()]

    for idx, tab in enumerate(tabs):
        with tab:
            if idx < len(st.session_state.chart_configs):
                config = st.session_state.chart_configs[idx]
                chart_type = st.radio(
                    "Select Chart Type",
                    CHART_OPTIONS,
                    index=CHART_OPTIONS.index(config["chart_type"]) if config["chart_type"] in CHART_OPTIONS else 0,
                    key=f"chart_type_{idx}"
                )
                st.session_state.chart_configs[idx]["chart_type"] = chart_type

                # Custom labels
                x_label = st.text_input("X-axis Label", x_axis if 'x_axis' in locals() else "", key=f"x_label_{idx}")
                y_label = st.text_input("Y-axis Label", y_axis if 'y_axis' in locals() else "", key=f"y_label_{idx}")

                # Interactive Chart Editing
                with st.expander("Edit Chart Properties"):
                    font_size = st.slider("Font Size", 8, 20, 12, key=f"font_size_{idx}")
                    legend_position = st.selectbox("Legend Position", LEGEND_POSITIONS, key=f"legend_pos_{idx}")
                    if chart_type == "Line Chart" and datetime_cols:
                        animation_col = st.selectbox("Animation Frame Column", datetime_cols, key=f"anim_col_{idx}")

                x_axis, y_axis, agg_func = None, None, None
                if chart_type in ["Scatter Plot", "Line Chart", "Violin Plot"]:
                    x_options = numeric_cols if chart_type == "Scatter Plot" else df.columns.tolist()
                    y_options = numeric_cols
                    x_axis = st.selectbox("Select X-axis", x_options, key=f"x_axis_{idx}")
                    y_axis = st.selectbox("Select Y-axis", y_options, key=f"y_axis_{idx}")
                elif chart_type == "Bar Chart":
                    x_axis = st.selectbox("Group by (X-axis)", categorical_cols, key=f"x_axis_{idx}")
                    y_axis = st.selectbox("Aggregate column (Y-axis)", numeric_cols, key=f"y_axis_{idx}")
                    agg_func = st.selectbox("Aggregation function", AGG_OPTIONS, key=f"agg_func_{idx}")
                elif chart_type == "Histogram":
                    x_axis = st.selectbox("Select column", numeric_cols, key=f"x_axis_{idx}")
                elif chart_type == "Box Plot":
                    y_axis = st.selectbox("Select column (Y-axis)", numeric_cols, key=f"y_axis_{idx}")
                    x_axis = st.selectbox("Optional: Group by (X-axis)", [None] + categorical_cols, key=f"x_axis_{idx}")
                elif chart_type == "Pie Chart":
                    x_axis = st.selectbox("Category", categorical_cols, key=f"x_axis_{idx}")
                    y_axis = st.selectbox("Values", numeric_cols, key=f"y_axis_{idx}")
                    agg_func = 'sum'  # Default for pie

                # --- Render Chart ---
                df_to_plot = df_filtered
                fig = None
                try:
                    if chart_type in ["Bar Chart", "Pie Chart"] and x_axis and y_axis and agg_func:
                        con = duckdb.connect()
                        con.register('df_to_plot', df_to_plot)
                        agg_query = f'SELECT "{x_axis}", {agg_func}("{y_axis}") AS agg_y FROM df_to_plot GROUP BY "{x_axis}"'
                        df_to_plot = con.execute(agg_query).fetchdf()

                    if chart_type == "Map View" and lat_col and lon_col:
                        st.map(df_filtered, latitude=lat_col, longitude=lon_col)
                    elif chart_type == "Correlation Heatmap" and len(df_filtered[numeric_cols].columns) >= 2:
                        corr_matrix = df_filtered[numeric_cols].corr()
                        fig, ax = plt.subplots(figsize=(12, 9))
                        sns.heatmap(corr_matrix, annot=True, cmap=color_palette.lower(), fmt=".2f", linewidths=.5,
                                    ax=ax)
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.tick_params(labelsize=font_size)
                        st.pyplot(fig)
                    else:
                        is_ready_to_plot = (x_axis and y_axis) or (chart_type == "Histogram" and x_axis) or (
                                chart_type in ["Box Plot", "Violin Plot"] and y_axis)
                        if not is_ready_to_plot:
                            st.warning("Please configure the required chart settings.")
                            continue

                        plot_args = {
                            'data_frame': df_to_plot,
                            'template': theme,
                            'color_discrete_sequence': px.colors.sequential.__getattribute__(color_palette),
                            'labels': {'x': x_label, 'y': y_label}
                        }
                        if chart_type == "Line Chart" and 'animation_col' in locals() and animation_col:
                            plot_args['animation_frame'] = animation_col
                        plot_functions = {
                            "Scatter Plot": px.scatter,
                            "Line Chart": px.line,
                            "Bar Chart": px.bar,
                            "Histogram": px.histogram,
                            "Box Plot": px.box,
                            "Violin Plot": px.violin,
                            "Pie Chart": px.pie
                        }
                        if chart_type in plot_functions:
                            plot_func = plot_functions[chart_type]
                            if chart_type == "Histogram":
                                fig = plot_func(**plot_args, x=x_axis, title=custom_title)
                            elif chart_type == "Pie Chart":
                                fig = plot_func(**plot_args, names=x_axis, values='agg_y', title=custom_title)
                            else:
                                fig = plot_func(**plot_args, x=x_axis,
                                                y=y_axis if chart_type != "Bar Chart" else 'agg_y', title=custom_title)
                            fig.update_layout(
                                font=dict(size=font_size),
                                legend=dict(orientation="h" if legend_position in ["top", "bottom"] else "v",
                                            yanchor="bottom" if legend_position == "bottom" else "top",
                                            y=-0.2 if legend_position == "bottom" else 1.02,
                                            xanchor="right" if legend_position == "right" else "left",
                                            x=1 if legend_position == "right" else 0) if legend_position != "none" else dict(
                                    showlegend=False)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    if fig:
                        figs.append(fig)
                except Exception as e:
                    st.error(f"Error rendering chart: {e}")
                    logging.error(f"Chart rendering error: {e}")

                # --- Export Options ---
                st.markdown("---")
                st.subheader("Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = convert_df_to_csv(df_to_plot)
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv_data,
                        file_name=f"{chart_type.lower().replace(' ', '_')}_data_{idx}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    if isinstance(fig, plt.Figure):
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        img_bytes = buf.getvalue()
                        st.download_button(
                            label="🖼️ Download Chart as PNG",
                            data=img_bytes,
                            file_name=f"{chart_type.lower().replace(' ', '_')}_chart_{idx}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    elif fig:
                        chart_json = fig.to_json()
                        st.download_button(
                            label="📊 Download Chart as JSON",
                            data=chart_json,
                            file_name=f"{chart_type.lower().replace(' ', '_')}_chart_{idx}.json",
                            mime="application/json",
                            use_container_width=True
                        )

    # --- Data Preview and Metadata ---
    with st.expander("Data Preview and Metadata"):
        st.write(f"**Data being plotted:** Showing {df_filtered.shape[0]} rows.")
        st.dataframe(df_filtered.head(10))
        st.write(f"**Columns**: {', '.join(df.columns)}")
        st.write(f"**Numeric Columns**: {', '.join(numeric_cols)}")
        st.write(f"**Categorical Columns**: {', '.join(categorical_cols)}")
        if datetime_cols:
            st.write(f"**Datetime Columns**: {', '.join(datetime_cols)}")

    # --- Collaboration Features: Save/Load Session State ---
    st.header("Collaboration Features")
    with st.expander("Save/Load Configuration"):
        if st.button("Save Session State"):
            session_json = json.dumps({
                "chart_configs": st.session_state.chart_configs,
                "filter_state": st.session_state.filter_state
            })
            st.download_button(
                label="Download Configuration JSON",
                data=session_json,
                file_name="session_config.json",
                mime="application/json"
            )
        uploaded_config = st.file_uploader("Upload Configuration JSON", type="json")
        if uploaded_config:
            config_data = json.load(uploaded_config)
            st.session_state.chart_configs = config_data.get("chart_configs", [])
            st.session_state.filter_state = config_data.get("filter_state", {})
            st.success("Configuration loaded! Rerun to apply.")

    # --- Dashboard Export ---
    st.header("Dashboard Export")
    if st.button("Export Dashboard to HTML"):
        html_content = generate_dashboard_html(df_filtered, eda_data,
                                               cluster_result if 'cluster_result' in locals() else None,
                                               rf_result if 'rf_result' in locals() else None, figs)
        st.download_button(
            label="Download Dashboard HTML",
            data=html_content,
            file_name="dashboard_export.html",
            mime="text/html"
        )

    # --- In-App Help ---
    with st.sidebar.expander("Help & Tutorials"):
        st.markdown("""
            **Welcome to the Advanced Data Visualization Platform!**
            - Upload files (batch) to start.
            - View EDA and clean data in the main area.
            - Run ML clustering (with auto-k suggestion, preprocessing) or Random Forest (with tuning, preprocessing) in the main area.
            - Configure filters and charts in the sidebar.
            - Visualize, edit properties (including animation), and export in the main area.
            - Save/load configurations for collaboration.
- Export full dashboard to HTML.
 """)


if __name__ == "__main__":

    main()
