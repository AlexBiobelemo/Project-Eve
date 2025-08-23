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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    mean_absolute_error, explained_variance_score, max_error, median_absolute_error,
    mean_absolute_percentage_error, balanced_accuracy_score, matthews_corrcoef, 
    cohen_kappa_score, roc_auc_score, average_precision_score, log_loss
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import RFE
import base64
import warnings
import joblib
import pickle
warnings.filterwarnings('ignore')

# --- Performance Configuration ---
st.set_page_config(
    page_title="Advanced Data Visualization Platform",
    page_icon="⚡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'eda_cache' not in st.session_state:
    st.session_state.eda_cache = {}
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {}

# --- Performance Optimized Helper Functions ---

@st.cache_data(show_spinner="Loading data...", max_entries=10)
def load_data(file_content: bytes, file_name: str, file_type: str) -> Optional[pd.DataFrame]:
    """Optimized data loading with caching."""
    try:
        if file_type == "CSV":
            df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
        elif file_type == "Excel":
            df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        elif file_type == "JSON":
            df = pd.read_json(io.BytesIO(file_content))
        else:
            raise ValueError("Unsupported file type")
        
        if df.empty:
            st.warning(f"File {file_name} is empty")
            return None
        
        # Optimize column names for SQL compatibility
        df.columns = [re.sub(r'[^\w]', '_', str(col)).strip('_') for col in df.columns]
        
        # Handle infinite values efficiently
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        return df
        
    except UnicodeDecodeError:
        try:
            if file_type == "CSV":
                df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')
                df.columns = [re.sub(r'[^\w]', '_', str(col)).strip('_') for col in df.columns]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
                return df
        except Exception as e2:
            logging.error(f"Data loading error with fallback encoding: {e2}")
            st.error(f"Failed to load file with multiple encodings: {e2}")
    except Exception as e:
        logging.error(f"Data loading error: {e}")
        st.error(f"Failed to load file: {e}")
        return None

@st.cache_data(show_spinner=False, max_entries=5)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Cached CSV conversion."""
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        logging.error(f"CSV conversion error: {e}")
        return b"Error converting data to CSV"

def find_lat_lon_columns(columns: pd.Index) -> Tuple[Optional[str], Optional[str]]:
    """Identifies latitude/longitude columns."""
    lat_col, lon_col = None, None
    lat_pattern = re.compile(r'.*(lat|latitude).*', re.IGNORECASE)
    lon_pattern = re.compile(r'.*(lon|lng|long|longitude).*', re.IGNORECASE)
    
    for col in columns:
        if not lat_col and lat_pattern.match(str(col)): 
            lat_col = col
        if not lon_col and lon_pattern.match(str(col)): 
            lon_col = col
    return lat_col, lon_col

def get_widget_value(param_name: str, default_value: Any, param_type: type = str) -> Any:
    """Gets widget value from URL params with type casting."""
    try:
        param_value = st.query_params.get(param_name)
        if param_value is None:
            return default_value
        
        if param_type in (list, tuple):
            return json.loads(param_value)
        return param_type(param_value)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return default_value

@st.cache_data(show_spinner=False)
def suggest_chart_type(df_shape: Tuple[int, int], num_numeric: int, num_categorical: int) -> List[str]:
    """Cached chart type suggestions based on data characteristics."""
    suggestions = []
    
    if num_numeric >= 2:
        suggestions.extend(["Correlation Heatmap", "Scatter Plot", "Box Plot"])
    if num_categorical > 0 and num_numeric > 0:
        suggestions.extend(["Bar Chart", "Box Plot"])
    if num_numeric > 0:
        suggestions.extend(["Histogram", "Box Plot"])
    if not suggestions:
        suggestions = ["Bar Chart"]
    
    return list(set(suggestions))

@st.cache_data(show_spinner="Computing EDA summary...", max_entries=5)
def compute_eda_summary(df_hash: str, df_shape: Tuple[int, int], sample_size: int = 50000) -> Dict[str, Any]:
    """Cached EDA computation using hash for cache invalidation."""
    try:
        # Get the actual dataframe from session state
        df = st.session_state.selected_df
        if df is None or df.empty:
            return {
                'shape': (0, 0),
                'describe': {},
                'dtypes': {},
                'missing': {},
                'insights': ['No data available for analysis']
            }
        
        # Sample large datasets for performance
        if len(df) > sample_size:
            sampled_df = df.sample(n=sample_size, random_state=42)
            sample_note = f" (analyzed {sample_size:,} sample rows)"
        else:
            sampled_df = df
            sample_note = ""
        
        numeric_cols = sampled_df.select_dtypes(include=['number']).columns.tolist()
        insights = []
        
        if numeric_cols:
            try:
                corr_matrix = sampled_df[numeric_cols].corr().fillna(0)
                high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
                
                corr_pairs = []
                for col1 in high_corr.columns:
                    for col2 in high_corr.index:
                        if high_corr.loc[col2, col1] and col1 < col2:  # Avoid duplicates
                            corr_val = corr_matrix.loc[col2, col1]
                            if not np.isnan(corr_val):
                                corr_pairs.append((col1, col2, corr_val))
                
                if corr_pairs:
                    # Show top 3 correlations
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    for col1, col2, corr_val in corr_pairs[:3]:
                        insights.append(f"High correlation: {col1} ↔ {col2} ({corr_val:.3f})")
                else:
                    insights.append("No strong correlations found" + sample_note)
                    
            except Exception as e:
                insights.append(f"Correlation analysis failed: {str(e)}")
        
        # Memory-efficient describe
        try:
            describe_data = sampled_df.describe(include='all').fillna('N/A').round(3).to_dict()
        except Exception:
            describe_data = {}
        
        # Additional insights
        missing_cols = sampled_df.isnull().sum()
        high_missing = missing_cols[missing_cols > len(sampled_df) * 0.5]
        if not high_missing.empty:
            insights.append(f"High missing data: {', '.join(high_missing.index[:3])} (>50%)")
        
        return {
            'shape': df_shape,
            'describe': describe_data,
            'dtypes': sampled_df.dtypes.astype(str).to_dict(),
            'missing': missing_cols.to_dict(),
            'insights': insights or ["Basic analysis completed" + sample_note],
            'sample_note': sample_note
        }
        
    except Exception as e:
        logging.error(f"EDA computation error: {e}")
        return {
            'shape': df_shape,
            'describe': {},
            'dtypes': {},
            'missing': {},
            'insights': [f'EDA analysis failed: {str(e)}'],
            'sample_note': ""
        }

def safe_sql_identifier(name: str) -> str:
    """Creates safe SQL identifiers."""
    safe_name = re.sub(r'[^\w]', '_', str(name))
    if safe_name and safe_name[0].isdigit():
        safe_name = f"col_{safe_name}"
    return safe_name or "unknown_column"

@st.cache_data(show_spinner="Applying filters...", max_entries=3)
def apply_duckdb_filters(df_hash: str, filter_state_str: str, categorical_cols: List[str],
                         numeric_cols: List[str], datetime_cols: List[str]) -> pd.DataFrame:
    """Cached filtering using DuckDB with performance optimization."""
    try:
        df = st.session_state.selected_df
        filter_state = json.loads(filter_state_str) if filter_state_str else {}
        
        if df.empty or not filter_state:
            return df
        
        # For large datasets, sample first then apply filters
        if len(df) > 1000000:  # 1M+ rows
            sample_df = df.sample(n=500000, random_state=42)
            st.warning("⚡ Using 500K sample for performance. Apply filters then use 'Process Full Dataset' button.")
        else:
            sample_df = df
        
        con = duckdb.connect()
        con.register('df_filtered', sample_df)
        query_parts = ["SELECT * FROM df_filtered"]
        conditions = []

        # Optimized categorical filters
        for col in categorical_cols[:10]:  # Limit for performance
            if f"filter_{col}" in filter_state and filter_state[f"filter_{col}"]:
                try:
                    safe_col = safe_sql_identifier(col)
                    values = filter_state[f"filter_{col}"][:50]  # Limit values
                    safe_values = []
                    for v in values:
                        if pd.isna(v) or str(v).lower() in ['nan', 'none', 'null']:
                            safe_values.append("NULL")
                        else:
                            escaped_val = str(v).replace("'", "''")
                            safe_values.append(f"'{escaped_val}'")
                    
                    if safe_values:
                        vals_str = ", ".join(safe_values)
                        if "NULL" in safe_values:
                            conditions.append(f'("{safe_col}" IN ({vals_str}) OR "{safe_col}" IS NULL)')
                        else:
                            conditions.append(f'"{safe_col}" IN ({vals_str})')
                except Exception as e:
                    logging.warning(f"Skipping categorical filter for {col}: {e}")

        # Optimized numeric filters
        for col in numeric_cols[:20]:  # Limit for performance
            if f"filter_{col}" in filter_state:
                try:
                    min_val, max_val = filter_state[f"filter_{col}"]
                    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                        continue
                    safe_col = safe_sql_identifier(col)
                    conditions.append(f'"{safe_col}" BETWEEN {min_val} AND {max_val}')
                except Exception as e:
                    logging.warning(f"Skipping numeric filter for {col}: {e}")

        # Datetime filters (limit to 3 for performance)
        for col in datetime_cols[:3]:
            if f"filter_{col}" in filter_state:
                try:
                    start_date, end_date = filter_state[f"filter_{col}"]
                    safe_col = safe_sql_identifier(col)
                    conditions.append(f'"{safe_col}" BETWEEN \'{start_date}\' AND \'{end_date}\'')
                except Exception as e:
                    logging.warning(f"Skipping datetime filter for {col}: {e}")

        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))

        query = " ".join(query_parts)
        result_df = con.execute(query).fetchdf()
        con.close()
        return result_df
        
    except Exception as e:
        logging.error(f"DuckDB filtering error: {e}")
        return st.session_state.selected_df

# --- Optimized ML Functions ---

@st.cache_data(show_spinner="Preprocessing data for clustering...", max_entries=3)
def preprocess_kmeans_cached(df_hash: str, numeric_cols: List[str], remove_outliers: bool = False) -> pd.DataFrame:
    """Cached K-means preprocessing."""
    try:
        df = st.session_state.selected_df
        if df.empty or not numeric_cols:
            return pd.DataFrame()
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        if not available_cols:
            return pd.DataFrame()
        
        data = df[available_cols].copy()
        data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        if data.empty or len(data) < 2:
            return pd.DataFrame()
        
        # Sample large datasets for performance
        if len(data) > 100000:
            data = data.sample(n=50000, random_state=42)
            st.info("⚡ Using 50K sample for clustering performance")
        
        if remove_outliers and len(data) > 10:
            try:
                iso = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso.fit_predict(data)
                data = data[outliers == 1]
            except Exception as e:
                logging.warning(f"Outlier removal failed: {e}")
        
        if data.empty or len(data) < 2:
            return pd.DataFrame()
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return pd.DataFrame(data_scaled, columns=available_cols, index=data.index)
        
    except Exception as e:
        logging.error(f"K-means preprocessing error: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner="Finding optimal clusters...", max_entries=2)
def optimal_k_cached(df_hash: str, numeric_cols: List[str], max_k: int = 10) -> int:
    """Cached optimal k computation."""
    try:
        data = preprocess_kmeans_cached(df_hash, numeric_cols)
        if data.empty or len(data) < 4:
            return 2
        
        max_k = min(max_k, len(data) - 1, 8)  # Limit for performance
        silhouettes = []
        
        for k in range(2, max_k + 1):
            try:
                # Use MiniBatch for large datasets
                if len(data) > 10000:
                    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                
                labels = kmeans.fit_predict(data)
                
                # Sample for silhouette score if too large
                if len(data) > 5000:
                    sample_idx = np.random.choice(len(data), 5000, replace=False)
                    score = silhouette_score(data.iloc[sample_idx], labels[sample_idx])
                else:
                    score = silhouette_score(data, labels)
                silhouettes.append(score)
            except Exception as e:
                logging.warning(f"Failed to compute silhouette for k={k}: {e}")
                silhouettes.append(-1)
        
        if not silhouettes or all(s <= 0 for s in silhouettes):
            return 3
        
        return silhouettes.index(max(silhouettes)) + 2
        
    except Exception as e:
        logging.error(f"Optimal k computation error: {e}")
        return 3

def perform_kmeans_optimized(df: pd.DataFrame, numeric_cols: List[str], k: int = 3, 
                           preprocess: bool = False, remove_outliers: bool = False) -> Dict[str, Any]:
    """Optimized K-Means clustering with session state caching."""
    
    model_key = f"kmeans_{hash(str(numeric_cols))}_{k}_{preprocess}_{remove_outliers}"
    
    # Check if model already exists in session state
    if model_key in st.session_state.trained_models:
        st.info("♻️ Using cached clustering model")
        return st.session_state.trained_models[model_key]
    
    try:
        if df.empty:
            return {"error": "Empty dataset provided"}
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        if not available_cols:
            return {"error": "None of the specified numeric columns exist"}
        
        if preprocess:
            df_hash = str(hash(str(df.values.tobytes())))
            data = preprocess_kmeans_cached(df_hash, available_cols, remove_outliers)
        else:
            data = df[available_cols].dropna()
        
        if data.empty or len(data) < k:
            return {"error": f"Insufficient data for clustering: {len(data)} samples < {k} clusters"}
        
        # Use appropriate algorithm based on data size
        if len(data) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1000)
            st.info("⚡ Using MiniBatch K-Means for performance")
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        # Time the clustering
        start_time = time.time()
        labels = kmeans.fit_predict(data)
        clustering_time = time.time() - start_time
        
        result = {
            "labels": labels, 
            "centers": kmeans.cluster_centers_,
            "inertia": kmeans.inertia_, 
            "index": data.index,
            "columns": available_cols,
            "clustering_time": clustering_time,
            "n_samples": len(data)
        }
        
        # Cache the result
        st.session_state.trained_models[model_key] = result
        return result
        
    except Exception as e:
        logging.error(f"K-means clustering error: {e}")
        return {"error": f"Clustering failed: {str(e)}"}

@st.cache_data(show_spinner="Preprocessing for Random Forest...", max_entries=2)
def preprocess_rf_cached(df_hash: str, x_cols: List[str], y_col: str, model_type: str, 
                        remove_outliers: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Cached Random Forest preprocessing."""
    try:
        df = st.session_state.selected_df
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        available_x_cols = [col for col in x_cols if col in df.columns]
        if not available_x_cols or y_col not in df.columns:
            raise ValueError("Required columns not found")
        
        # Sample large datasets
        if len(df) > 100000:
            sample_df = df.sample(n=50000, random_state=42)
            st.info("⚡ Using 50K sample for model training performance")
        else:
            sample_df = df
        
        X = sample_df[available_x_cols].copy()
        y = sample_df[y_col].copy()
        
        # Remove rows with NaN targets
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if X.empty:
            raise ValueError("No valid data after preprocessing")
        
        # Efficient outlier removal
        if remove_outliers and len(X) > 100:
            numeric_x_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_x_cols) > 0:
                try:
                    # Use smaller contamination for performance
                    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=50)
                    outliers = iso.fit_predict(X[numeric_x_cols])
                    X = X[outliers == 1]
                    y = y[outliers == 1]
                except Exception as e:
                    logging.warning(f"Outlier removal failed: {e}")
        
        # Efficient categorical encoding
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                # Use label encoding for efficiency
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill remaining NaN values
        X = X.fillna(X.mean(numeric_only=True)).fillna(0)
        
        # Feature selection for high-dimensional data
        if len(X.columns) > 50:
            try:
                from sklearn.feature_selection import SelectKBest, f_regression, f_classif
                selector = SelectKBest(
                    score_func=f_regression if model_type == "regression" else f_classif,
                    k=min(20, len(X.columns))
                )
                X_selected = selector.fit_transform(X, y)
                X = pd.DataFrame(X_selected, index=X.index)
                st.info(f"⚡ Selected top {X.shape[1]} features for performance")
            except Exception as e:
                logging.warning(f"Feature selection failed: {e}")
        
        return X, y
        
    except Exception as e:
        logging.error(f"Random Forest preprocessing error: {e}")
        raise e

def perform_random_forest_optimized(df: pd.DataFrame, x_cols: List[str], y_col: str, model_type: str, 
                                  preprocess: bool = False, tune: bool = False, 
                                  remove_outliers: bool = False) -> Dict[str, Any]:
    """Optimized Random Forest with session state caching."""
    
    model_key = f"rf_{hash(str(x_cols))}_{y_col}_{model_type}_{preprocess}_{tune}_{remove_outliers}"
    
    # Check if model exists in session state
    if model_key in st.session_state.trained_models:
        st.info("♻️ Using cached Random Forest model")
        return st.session_state.trained_models[model_key]
    
    try:
        start_time = time.time()
        
        if df.empty:
            return {"error": "Empty dataset provided"}
        
        df_hash = str(hash(str(df.values.tobytes())))
        
        if preprocess:
            X, y = preprocess_rf_cached(df_hash, x_cols, y_col, model_type, remove_outliers)
        else:
            available_x_cols = [col for col in x_cols if col in df.columns]
            if not available_x_cols or y_col not in df.columns:
                return {"error": "Required columns not found"}
            
            X = df[available_x_cols].dropna()
            y = df[y_col].dropna()
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
        
        if len(X) != len(y) or len(X) < 10:
            return {"error": f"Insufficient data: {len(X)} samples"}
        
        # Check target for classification
        if model_type == "classification" and len(y.unique()) < 2:
            return {"error": "Classification requires at least 2 different target values"}
        
        # Optimized train-test split
        test_size = min(0.3, max(0.1, 1000 / len(X)))  # Adaptive but capped
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Use optimized model parameters
        base_params = {
            'random_state': 42,
            'n_estimators': 50,  # Reduced for speed
            'max_depth': 10,     # Limit depth
            'min_samples_split': 5,
            'n_jobs': -1
        }
        
        if tune:
            # Quick hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
            
            scoring = 'r2' if model_type == "regression" else 'f1_weighted'
            model_class = RandomForestRegressor if model_type == "regression" else RandomForestClassifier
            
            search = RandomizedSearchCV(
                model_class(**base_params), 
                param_grid, 
                n_iter=6,  # Limited iterations
                cv=3,      # Reduced folds
                random_state=42, 
                scoring=scoring,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params = search.best_params_
            cv_mean = search.best_score_
        else:
            model = RandomForestRegressor(**base_params) if model_type == "regression" else RandomForestClassifier(**base_params)
            model.fit(X_train, y_train)
            best_params = {}
            cv_mean = 0.0
        
        # Predictions and evaluation
        y_pred = model.predict(X_test)
        
        # Get probabilities for classification
        y_proba = None
        if model_type == "classification" and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Enhanced metrics calculation
        metrics = evaluate_model_enhanced(y_test, y_pred, model_type, y_proba)
        
        training_time = time.time() - start_time
        
        result = {
            "model": model,
            "metrics": metrics,
            "feature_importances": getattr(model, 'feature_importances_', None),
            "feature_names": list(X.columns) if hasattr(X, 'columns') else None,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "best_params": best_params,
            "cv_mean": cv_mean,
            "training_time": training_time,
            "n_samples": len(X)
        }
        
        # Cache the result
        st.session_state.trained_models[model_key] = result
        return result
        
    except Exception as e:
        logging.error(f"Random Forest error: {e}")
        return {"error": f"Model training failed: {str(e)}"}

# --- Optimized Visualization Functions ---

@st.cache_data(show_spinner="Generating visualization...", max_entries=5)
def create_plotly_chart(chart_type: str, df_hash: str, x_axis: str, y_axis: str, 
                       color_col: str, title: str, theme: str) -> Any:
    """Cached chart generation."""
    try:
        df = st.session_state.selected_df
        if df.empty:
            return None
        
        # Sample large datasets for visualization performance
        if len(df) > 10000:
            plot_df = df.sample(n=10000, random_state=42)
            sample_note = f" (showing {len(plot_df):,} sample points)"
        else:
            plot_df = df
            sample_note = ""
        
        plot_args = {
            'data_frame': plot_df,
            'title': title + sample_note,
            'template': theme,
        }
        
        if color_col and color_col in plot_df.columns:
            plot_args['color'] = color_col
        
        # Create chart based on type
        if chart_type == "Scatter Plot":
            fig = px.scatter(x=x_axis, y=y_axis, **plot_args)
        elif chart_type == "Line Chart":
            fig = px.line(x=x_axis, y=y_axis, **plot_args)
        elif chart_type == "Bar Chart":
            fig = px.bar(x=x_axis, y=y_axis, **plot_args)
        elif chart_type == "Histogram":
            fig = px.histogram(x=x_axis, **plot_args)
        elif chart_type == "Box Plot":
            fig = px.box(x=x_axis, y=y_axis, **plot_args)
        elif chart_type == "Violin Plot":
            fig = px.violin(x=x_axis, y=y_axis, **plot_args)
        elif chart_type == "Pie Chart":
            fig = px.pie(names=x_axis, values=y_axis, **plot_args)
        else:
            return None
        
        # Optimize layout for performance
        fig.update_layout(
            font=dict(size=12),
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logging.error(f"Chart creation error: {e}")
        return None

# --- Enhanced Metrics Functions ---

def enhanced_clustering_metrics(X, labels, n_clusters):
    """Comprehensive clustering evaluation metrics."""
    if len(np.unique(labels)) < 2:
        return {"error": "Need at least 2 clusters for metrics"}
    
    try:
        metrics = {}
        
        # Core clustering metrics
        metrics['silhouette_score'] = silhouette_score(X, labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels) 
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        
        # Cluster distribution analysis
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels, counts))
        metrics['cluster_balance'] = np.std(counts) / np.mean(counts)  # Lower = more balanced
        
        # Within-cluster sum of squares
        wcss = 0
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                cluster_center = cluster_points.mean(axis=0)
                wcss += ((cluster_points - cluster_center) ** 2).sum()
        metrics['wcss'] = wcss
        
        # Variance ratio  
        overall_center = X.mean(axis=0)
        bcss = 0
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                cluster_center = cluster_points.mean(axis=0)
                bcss += len(cluster_points) * ((cluster_center - overall_center) ** 2).sum()
        
        metrics['variance_ratio'] = bcss / (wcss + bcss) if (wcss + bcss) > 0 else 0
        
        return metrics
        
    except Exception as e:
        return {"error": f"Metrics computation failed: {str(e)}"}

def enhanced_regression_metrics(y_true, y_pred):
    """Comprehensive regression evaluation metrics."""
    try:
        metrics = {}
        
        # Standard metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # Robust metrics
        metrics['median_ae'] = median_absolute_error(y_true, y_pred)
        metrics['max_error'] = max_error(y_true, y_pred)
        
        # Percentage metrics (avoid division by zero)
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            metrics['mape'] = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
        else:
            metrics['mape'] = np.nan
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        
        # Prediction coverage
        metrics['coverage_80pct'] = np.mean(np.abs(residuals) <= 1.28 * metrics['residual_std'])
        metrics['coverage_95pct'] = np.mean(np.abs(residuals) <= 1.96 * metrics['residual_std'])
        
        return metrics
        
    except Exception as e:
        return {"error": f"Regression metrics failed: {str(e)}"}

def enhanced_classification_metrics(y_true, y_pred, y_proba=None):
    """Comprehensive classification evaluation metrics."""
    try:
        metrics = {}
        
        # Standard metrics
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, average='weighted', zero_division=0) 
        metrics["F1"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Advanced metrics
        metrics['Balanced_Accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        metrics['Cohen_Kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics (if available)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['PR_AUC'] = average_precision_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                
                metrics['Log_Loss'] = log_loss(y_true, y_proba)
                
            except Exception:
                pass  # Skip probability metrics if they fail
        
        return metrics
        
    except Exception as e:
        return {"error": f"Classification metrics failed: {str(e)}"}

def evaluate_model_enhanced(y_true: pd.Series, y_pred: np.ndarray, model_type: str, y_proba=None) -> Dict[str, float]:
    """Enhanced model evaluation with comprehensive metrics."""
    try:
        if model_type == "regression":
            return enhanced_regression_metrics(y_true, y_pred)
        else:
            return enhanced_classification_metrics(y_true, y_pred, y_proba)
    except Exception as e:
        logging.error(f"Enhanced model evaluation error: {e}")
        return {"Error": 0.0}
    """Times the execution of a function."""
    try:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    except Exception as e:
        return {"error": str(e)}, 0.0

def generate_dashboard_html(df: pd.DataFrame, eda_data: Dict[str, Any], 
                          cluster_result: Optional[Dict[str, Any]],
                          rf_result: Optional[Dict[str, Any]], 
                          figs: List[Any]) -> str:
    """Generates optimized HTML for dashboard export."""
    try:
        html_parts = [
            "<html><head>",
            "<title>Advanced Data Dashboard</title>",
            "<meta charset='utf-8'>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #2E86C1; }",
            "img { max-width: 100%; height: auto; margin: 10px 0; }",
            ".metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }",
            "</style>",
            "</head><body>",
            "<h1>📊 Data Analysis Dashboard</h1>",
            f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # EDA Section
        html_parts.extend([
            "<h2>📈 Exploratory Data Analysis</h2>",
            f"<div class='metric'><strong>Rows:</strong> {eda_data.get('shape', ['N/A', 'N/A'])[0]:,}</div>",
            f"<div class='metric'><strong>Columns:</strong> {eda_data.get('shape', ['N/A', 'N/A'])[1]}</div>",
        ])
        
        # Insights
        insights = eda_data.get('insights', ['No insights available'])
        html_parts.extend([
            "<h3>💡 Key Insights</h3>",
            "<ul>"
        ])
        for insight in insights[:5]:  # Limit for performance
            html_parts.append(f"<li>{insight}</li>")
        html_parts.append("</ul>")
        
        # ML Results
        if cluster_result and "error" not in cluster_result:
            html_parts.extend([
                "<h2>🎯 Clustering Analysis</h2>",
                f"<p><strong>Inertia:</strong> {cluster_result.get('inertia', 'N/A'):.2f}</p>",
                f"<p><strong>Samples:</strong> {cluster_result.get('n_samples', 'N/A'):,}</p>",
                f"<p><strong>Time:</strong> {cluster_result.get('clustering_time', 0):.2f}s</p>"
            ])
        
        if rf_result and "error" not in rf_result:
            metrics = rf_result.get('metrics', {})
            metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            html_parts.extend([
                "<h2>🌳 Random Forest Analysis</h2>",
                f"<p><strong>Metrics:</strong> {metrics_str}</p>",
                f"<p><strong>Samples:</strong> {rf_result.get('n_samples', 'N/A'):,}</p>",
                f"<p><strong>Training Time:</strong> {rf_result.get('training_time', 0):.2f}s</p>"
            ])
        
        # Charts (limit for performance)
        html_parts.append("<h2>📊 Visualizations</h2>")
        for i, fig in enumerate(figs[:5]):  # Limit to 5 charts
            if fig is not None:
                try:
                    if hasattr(fig, 'to_image'):
                        img_bytes = fig.to_image(format="png", width=800, height=500)
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Chart {i+1}"/>')
                    elif hasattr(fig, 'savefig'):
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Chart {i+1}"/>')
                except Exception as e:
                    html_parts.append(f"<p>Chart {i+1} export failed: {str(e)}</p>")
        
        html_parts.extend([
            f"<hr><p><small>Generated by Advanced Data Visualization Platform</small></p>",
            "</body></html>"
        ])
        
        return "".join(html_parts)
        
    except Exception as e:
        logging.error(f"Dashboard HTML generation error: {e}")
        return f"<html><body><h1>Dashboard Export Error</h1><p>{str(e)}</p></body></html>"

# --- Main Application ---

def main() -> None:
    """Optimized main function for the Streamlit app."""
    try:
        # --- Apply Custom CSS ---
        st.markdown("""
            <style>
            .stApp { font-family: 'Segoe UI', Arial, sans-serif; }
            .metric-card { 
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;
            }
            .performance-badge {
                background: #28a745; color: white; padding: 0.25rem 0.5rem; 
                border-radius: 15px; font-size: 0.8rem; margin-left: 0.5rem;
            }
            @media (max-width: 600px) {
                .stSidebar { width: 100% !important; }
                .stButton > button { width: 100%; }
            }
            </style>
        """, unsafe_allow_html=True)

        st.title("⚡️ Advanced Data Visualization Platform")
        st.markdown("*Powered by optimized caching and smart performance features*")

        # Performance monitoring
        if 'app_start_time' not in st.session_state:
            st.session_state.app_start_time = time.time()

        # --- File Upload with Progress ---
        file_type = st.selectbox("File Type", FILE_TYPES, help="Select file format")
        uploaded_files = st.file_uploader("Upload your files (batch supported)", 
                                        type=['csv', 'xlsx', 'json', 'xls'],
                                        accept_multiple_files=True,
                                        help="Upload multiple files for analysis")

        if not uploaded_files:
            st.info("Please upload files to begin.")
            
            # Show performance tips
            with st.expander("⚡ Performance Tips"):
                st.markdown("""
                **🚀 This app is optimized for speed:**
                - **Smart Caching**: Data loads only once
                - **Sampling**: Large datasets auto-sampled for speed  
                - **Lazy Loading**: Heavy computations run on-demand
                - **Session State**: Models cached between interactions
                
                **📊 Best Practices:**
                - Start with smaller datasets (< 100K rows) for exploration
                - Use filters to focus on relevant data subsets
                - Enable preprocessing for mixed data types
                - Save sessions for collaboration
                """)
            
            # Quick demo option
            if st.button("🎯 Load Demo Dataset", key="demo_btn"):
                # Create demo data
                np.random.seed(42)
                demo_data = pd.DataFrame({
                    'sales': np.random.normal(1000, 200, 5000),
                    'profit': np.random.normal(150, 50, 5000),
                    'category': np.random.choice(['A', 'B', 'C', 'D'], 5000),
                    'region': np.random.choice(['North', 'South', 'East', 'West'], 5000),
                    'date': pd.date_range('2023-01-01', periods=5000, freq='H')
                })
                st.session_state.dfs['demo_data.csv'] = demo_data
                st.session_state.selected_df = demo_data
                st.session_state.data_loaded = True
                st.success("✅ Demo dataset loaded! (5,000 rows)")
                st.rerun()
            
            return

        # --- Optimized File Processing ---
        current_file_names = [f.name for f in uploaded_files]
        if st.session_state.last_uploaded_files != current_file_names:
            st.session_state.chart_configs = []
            st.session_state.filter_state = {}
            st.session_state.last_uploaded_files = current_file_names
            st.session_state.dfs = {}
            st.session_state.trained_models = {}  # Clear model cache

        # Process files with progress and performance tracking
        with st.spinner("Processing files with optimized loading..."):
            progress_bar = st.progress(0)
            load_times = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                start_time = time.time()
                
                # Use file content for caching
                file_content = uploaded_file.read()
                uploaded_file.seek(0)  # Reset for potential re-reading
                
                df = load_data(file_content, uploaded_file.name, file_type)
                load_time = time.time() - start_time
                load_times.append(load_time)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                if df is not None:
                    st.session_state.dfs[uploaded_file.name] = df
                    
                    # Show performance metrics
                    speed_mb_s = (len(file_content) / 1024 / 1024) / load_time if load_time > 0 else 0
                    st.success(f"✅ {uploaded_file.name}: {df.shape[0]:,} × {df.shape[1]} ({speed_mb_s:.1f} MB/s)")
                else:
                    st.error(f"❌ Failed to load {uploaded_file.name}")

            # Show overall performance
            if load_times:
                avg_time = np.mean(load_times)
                total_rows = sum(len(df) for df in st.session_state.dfs.values())
                st.info(f"⚡ Processed {len(uploaded_files)} files, {total_rows:,} total rows in {avg_time:.2f}s average")

        if not st.session_state.dfs:
            st.error("No files were successfully loaded. Please check your file formats.")
            return

        # --- Dataset Selection ---
        selected_file = st.selectbox(
            "Select Dataset for Analysis", 
            list(st.session_state.dfs.keys()),
            help="Choose which dataset to analyze"
        )
        df = st.session_state.dfs[selected_file]
        st.session_state.selected_df = df

        if df.empty:
            st.error("The selected dataset is empty.")
            return

        st.session_state.data_loaded = True
        
        # --- Dataset Analysis with Caching ---
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            lat_col, lon_col = find_lat_lon_columns(df.columns)
        except Exception as e:
            st.error(f"Error analyzing dataset structure: {e}")
            return

        # --- Optimized EDA Section ---
        st.header("📊 Exploratory Data Analysis")
        
        # Create a hash for caching
        df_hash = str(hash(str(df.shape) + str(df.columns.tolist())))
        
        try:
            eda_data = compute_eda_summary(df_hash, df.shape)
            
            # Performance-optimized metrics display
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Rows", f"{eda_data['shape'][0]:,}")
            with col2:
                st.metric("Columns", f"{eda_data['shape'][1]}")
            with col3:
                st.metric("Numeric", len(numeric_cols))
            with col4:
                st.metric("Categorical", len(categorical_cols))
            with col5:
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory", f"{memory_mb:.1f} MB")
            
            # Expandable sections for details
            col_left, col_right = st.columns(2)
            
            with col_left:
                with st.expander("📈 Statistical Summary"):
                    if eda_data['describe']:
                        try:
                            # Show only key statistics for performance
                            describe_df = pd.DataFrame(eda_data['describe'])
                            if len(describe_df.columns) > 10:
                                describe_df = describe_df.iloc[:, :10]
                                st.info("Showing first 10 columns for performance")
                            st.dataframe(describe_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not display summary: {e}")
            
            with col_right:
                with st.expander("💡 Key Insights"):
                    for insight in eda_data['insights']:
                        st.info(insight)
            
            # Quick data quality check
            with st.expander("🔍 Data Quality Check"):
                quality_issues = []
                
                # Missing data check
                missing_pct = (df.isnull().sum() / len(df)) * 100
                high_missing = missing_pct[missing_pct > 50]
                if not high_missing.empty:
                    quality_issues.append(f"High missing data (>50%): {', '.join(high_missing.index[:3])}")
                
                # Duplicate rows
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    quality_issues.append(f"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")
                
                # Data type issues
                mixed_types = []
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if column contains mixed types
                        sample_vals = df[col].dropna().head(100)
                        types_found = set(type(val).__name__ for val in sample_vals)
                        if len(types_found) > 1:
                            mixed_types.append(col)
                
                if mixed_types:
                    quality_issues.append(f"Mixed data types: {', '.join(mixed_types[:3])}")
                
                if quality_issues:
                    for issue in quality_issues:
                        st.warning(issue)
                else:
                    st.success("✅ No major data quality issues detected")
            
        except Exception as e:
            st.error(f"EDA analysis failed: {e}")

        # --- Performance-Optimized Sidebar ---
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # Performance monitor
            if st.session_state.trained_models:
                st.success(f"🎯 {len(st.session_state.trained_models)} cached models")
            
            # Clear cache option
            if st.button("🗑️ Clear Cache", key="clear_cache_btn"):
                st.session_state.trained_models = {}
                st.session_state.eda_cache = {}
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")

            # --- Smart Filtering ---
            st.subheader("🔍 Smart Data Filtering")
            with st.expander("Apply Filters", expanded=True):
                filter_applied = False
                
                # Limit filters for performance
                if categorical_cols:
                    st.write("**Categorical (top 3):**")
                    for col in categorical_cols[:3]:
                        try:
                            unique_vals = df[col].dropna().unique()
                            if len(unique_vals) > 50:
                                st.info(f"{col}: {len(unique_vals)} values (showing top 20)")
                                unique_vals = sorted(unique_vals)[:20]
                            
                            selected = st.multiselect(
                                col, 
                                unique_vals, 
                                key=f"filter_{col}",
                                max_selections=10  # Limit for performance
                            )
                            if selected:
                                st.session_state.filter_state[f"filter_{col}"] = selected
                                filter_applied = True
                        except Exception as e:
                            st.warning(f"Filter error for {col}: {e}")

                # Numeric filters (limit to top 5)
                if numeric_cols:
                    st.write("**Numeric (top 5):**")
                    for col in numeric_cols[:5]:
                        try:
                            col_data = df[col].dropna()
                            if len(col_data) > 0:
                                min_val, max_val = float(col_data.min()), float(col_data.max())
                                if min_val != max_val:
                                    range_val = st.slider(
                                        col, min_val, max_val, (min_val, max_val),
                                        key=f"range_{col}",
                                        format="%.2f"
                                    )
                                    if range_val != (min_val, max_val):
                                        st.session_state.filter_state[f"filter_{col}"] = range_val
                                        filter_applied = True
                        except Exception as e:
                            st.warning(f"Numeric filter error for {col}: {e}")

            # --- Chart Configuration ---
            st.subheader("📊 Chart Configuration")
            
            # Smart suggestions
            suggested_charts = suggest_chart_type(df.shape, len(numeric_cols), len(categorical_cols))
            if suggested_charts:
                st.info(f"💡 Suggested: {', '.join(suggested_charts[:2])}")

            # Chart management
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➕ Add Chart", key="add_new_chart_btn"):
                    st.session_state.chart_configs.append({
                        "chart_type": suggested_charts[0] if suggested_charts else CHART_OPTIONS[0], 
                        "id": len(st.session_state.chart_configs)
                    })
                    st.rerun()
            
            with col2:
                if st.session_state.chart_configs and st.button("🗑️ Clear Charts", key="clear_charts_btn"):
                    st.session_state.chart_configs = []
                    st.rerun()

        # --- Apply Filters with Performance Monitoring ---
        if st.session_state.filter_state:
            filter_state_str = json.dumps(st.session_state.filter_state, default=str)
            df_filtered = apply_duckdb_filters(
                df_hash, filter_state_str, categorical_cols, numeric_cols, datetime_cols
            )
            
            if len(df_filtered) != len(df):
                reduction_pct = (1 - len(df_filtered) / len(df)) * 100
                st.info(f"🔍 Filtered: {len(df_filtered):,} rows ({reduction_pct:.1f}% reduction)")
        else:
            df_filtered = df

        # --- Machine Learning Section ---
        st.header("🤖 Machine Learning")
        
        ml_col1, ml_col2 = st.columns(2)
        
        with ml_col1:
            # K-Means Clustering
            with st.expander("🎯 K-Means Clustering"):
                if not numeric_cols:
                    st.warning("No numeric columns for clustering")
                else:
                    # Configuration
                    auto_k = st.checkbox("Auto-suggest k", key="auto_k_check")
                    remove_outliers_kmeans = st.checkbox("Remove outliers", key="kmeans_outliers")
                    preprocess_kmeans_flag = st.checkbox("Preprocess data", key="kmeans_preprocess")
                    
                    selected_cols = st.multiselect(
                        "Features", numeric_cols, 
                        default=numeric_cols[:3],
                        key="kmeans_features"
                    )
                    
                    if auto_k and selected_cols:
                        try:
                            k = optimal_k_cached(df_hash, selected_cols)
                            st.info(f"🎯 Suggested k: {k}")
                        except:
                            k = 3
                    else:
                        k = st.slider("Number of clusters", 2, 8, 3, key="k_slider")
                    
                    # Training button with unique key
                    if st.button("🚀 Run Clustering", key="run_clustering_btn") and selected_cols:
                        with st.spinner("Running optimized K-Means..."):
                            cluster_result = perform_kmeans_optimized(
                                df_filtered, selected_cols, k,
                                preprocess=preprocess_kmeans_flag,
                                remove_outliers=remove_outliers_kmeans
                            )
                        
                        if "error" in cluster_result:
                            st.error(f"❌ {cluster_result['error']}")
                        else:
                            st.success(f"✅ Clustering completed in {cluster_result.get('clustering_time', 0):.2f}s!")
                            
                            # Add results to dataframe
                            df_filtered_copy = df_filtered.copy()
                            df_filtered_copy['cluster'] = -1
                            df_filtered_copy.loc[cluster_result["index"], 'cluster'] = cluster_result["labels"]
                            
                            # Enhanced clustering metrics
                            if preprocess_kmeans_flag:
                                data_for_metrics = preprocess_kmeans_cached(df_hash, selected_cols, remove_outliers_kmeans)
                            else:
                                data_for_metrics = df_filtered[selected_cols].dropna()
                            
                            if not data_for_metrics.empty and len(data_for_metrics) == len(cluster_result["labels"]):
                                enhanced_metrics = enhanced_clustering_metrics(
                                    data_for_metrics.values, 
                                    cluster_result["labels"], 
                                    k
                                )
                                
                                if "error" not in enhanced_metrics:
                                    # Display comprehensive metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Silhouette Score", f"{enhanced_metrics['silhouette_score']:.4f}", 
                                                help="0-1, higher = better separation")
                                    with col2:
                                        st.metric("Davies-Bouldin", f"{enhanced_metrics['davies_bouldin_score']:.4f}",
                                                help="Lower = better clustering")
                                    with col3:
                                        st.metric("Calinski-Harabasz", f"{enhanced_metrics['calinski_harabasz_score']:.0f}",
                                                help="Higher = better clustering")
                                    with col4:
                                        st.metric("Variance Ratio", f"{enhanced_metrics['variance_ratio']:.4f}",
                                                help="0-1, higher = better")
                                    
                                    # Additional cluster analysis
                                    with st.expander("📊 Detailed Cluster Analysis"):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.write("**Cluster Sizes:**")
                                            cluster_df = pd.DataFrame(
                                                list(enhanced_metrics['cluster_sizes'].items()), 
                                                columns=['Cluster', 'Size']
                                            )
                                            st.dataframe(cluster_df, use_container_width=True)
                                        
                                        with col_b:
                                            st.metric("Cluster Balance", f"{enhanced_metrics['cluster_balance']:.4f}",
                                                    help="Lower = more balanced")
                                            st.metric("WCSS", f"{enhanced_metrics['wcss']:.2f}",
                                                    help="Within-cluster sum of squares")
                            
                            # Show quick visualization
                            if len(selected_cols) >= 2:
                                fig = px.scatter(
                                    df_filtered_copy[df_filtered_copy['cluster'] != -1], 
                                    x=selected_cols[0], y=selected_cols[1],
                                    color='cluster', 
                                    title=f"K-Means Clusters (k={k})",
                                    color_discrete_sequence=px.colors.qualitative.Set1
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Basic performance metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Inertia", f"{cluster_result['inertia']:.2f}")
                            with col2:
                                st.metric("Samples", f"{cluster_result['n_samples']:,}")
                            with col3:
                                st.metric("Time", f"{cluster_result['clustering_time']:.2f}s")

        with ml_col2:
            # Random Forest
            with st.expander("🌳 Random Forest"):
                if not (numeric_cols or categorical_cols):
                    st.warning("No suitable features for modeling")
                else:
                    # Configuration
                    rf_type = st.selectbox("Model type", ["regression", "classification"], key="rf_type")
                    tune_rf = st.checkbox("Hyperparameter tuning", key="rf_tune")
                    preprocess_rf = st.checkbox("Auto preprocessing", key="rf_preprocess")
                    
                    # Feature selection
                    available_features = numeric_cols + categorical_cols
                    x_features = st.multiselect(
                        "Features (X)", available_features,
                        default=numeric_cols[:3],
                        key="rf_features"
                    )
                    
                    # Target selection
                    target_options = [col for col in df_filtered.columns if col not in x_features]
                    y_target = st.selectbox("Target (Y)", target_options, key="rf_target")
                    
                    # Training button with unique key
                    if st.button("🚀 Train Model", key="train_model_btn") and x_features and y_target:
                        with st.spinner("Training optimized Random Forest..."):
                            rf_result = perform_random_forest_optimized(
                                df_filtered, x_features, y_target, rf_type,
                                preprocess=preprocess_rf, tune=tune_rf
                            )
                        
                        if "error" in rf_result:
                            st.error(f"❌ {rf_result['error']}")
                        else:
                            st.success(f"✅ Model trained in {rf_result.get('training_time', 0):.2f}s!")
                            
                            # Enhanced metrics display
                            metrics = rf_result.get('metrics', {})
                            if metrics and "error" not in metrics:
                                if rf_type == "regression":
                                    # Regression metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                                    with col2:
                                        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                                    with col3:
                                        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                                    with col4:
                                        mape_val = metrics.get('mape', np.nan)
                                        st.metric("MAPE", f"{mape_val:.2%}" if not np.isnan(mape_val) else "N/A")
                                    
                                    # Advanced regression metrics
                                    with st.expander("📈 Advanced Regression Metrics"):
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            st.write("**Error Analysis:**")
                                            st.metric("Max Error", f"{metrics.get('max_error', 0):.4f}")
                                            st.metric("Median AE", f"{metrics.get('median_ae', 0):.4f}")
                                            st.metric("Explained Var", f"{metrics.get('explained_variance', 0):.4f}")
                                        
                                        with col_b:
                                            st.write("**Residual Stats:**")
                                            st.metric("Residual Mean", f"{metrics.get('residual_mean', 0):.6f}")
                                            st.metric("Residual Std", f"{metrics.get('residual_std', 0):.4f}")
                                        
                                        with col_c:
                                            st.write("**Prediction Quality:**")
                                            st.metric("80% Coverage", f"{metrics.get('coverage_80pct', 0):.2%}")
                                            st.metric("95% Coverage", f"{metrics.get('coverage_95pct', 0):.2%}")
                                
                                else:  # Classification
                                    # Classification metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
                                    with col2:
                                        st.metric("F1 Score", f"{metrics.get('F1', 0):.4f}")
                                    with col3:
                                        st.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
                                    with col4:
                                        st.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
                                    
                                    # Advanced classification metrics
                                    with st.expander("🎯 Advanced Classification Metrics"):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.write("**Balanced Metrics:**")
                                            st.metric("Balanced Accuracy", f"{metrics.get('Balanced_Accuracy', 0):.4f}")
                                            st.metric("Matthews Corr Coef", f"{metrics.get('MCC', 0):.4f}")
                                            st.metric("Cohen's Kappa", f"{metrics.get('Cohen_Kappa', 0):.4f}")
                                        
                                        with col_b:
                                            st.write("**Probability Metrics:**")
                                            if 'ROC_AUC' in metrics:
                                                st.metric("ROC AUC", f"{metrics['ROC_AUC']:.4f}")
                                            if 'PR_AUC' in metrics:
                                                st.metric("PR AUC", f"{metrics['PR_AUC']:.4f}")
                                            if 'Log_Loss' in metrics:
                                                st.metric("Log Loss", f"{metrics['Log_Loss']:.4f}")
                            
                            # Cross-validation results
                            if rf_result.get('cv_mean', 0) != 0:
                                st.write("🔄 **Cross-Validation Results:**")
                                cv_cols = st.columns(3)
                                with cv_cols[0]:
                                    st.metric("CV Mean", f"{rf_result['cv_mean']:.4f}")
                                with cv_cols[1]:
                                    st.metric("CV Std", f"{rf_result.get('cv_std', 0):.4f}")
                                with cv_cols[2]:
                                    st.metric("CV Min", f"{rf_result.get('cv_min', 0):.4f}")
                            
                            # Best parameters
                            if rf_result.get("best_params"):
                                st.write("⚙️ **Optimized Parameters:**")
                                st.json(rf_result['best_params'])
                            
                            # Feature importance
                            if rf_result.get("feature_importances") is not None and rf_result.get("feature_names"):
                                try:
                                    importance_df = pd.DataFrame({
                                        'Feature': rf_result['feature_names'],
                                        'Importance': rf_result['feature_importances']
                                    }).sort_values('Importance', ascending=False)
                                    
                                    fig = px.bar(
                                        importance_df.head(10),
                                        x='Importance', y='Feature',
                                        orientation='h',
                                        title="Top 10 Feature Importances"
                                    )
                                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not display feature importance: {e}")
                            
                            # Prediction vs Actual plot
                            if 'y_test' in rf_result and 'y_pred' in rf_result:
                                try:
                                    pred_df = pd.DataFrame({
                                        'Actual': rf_result['y_test'],
                                        'Predicted': rf_result['y_pred']
                                    })
                                    
                                    fig_pred = px.scatter(
                                        pred_df, x='Actual', y='Predicted',
                                        title="Actual vs Predicted Values",
                                        trendline="ols" if rf_type == "regression" else None
                                    )
                                    
                                    # Add perfect prediction line
                                    min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                                    max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                                    fig_pred.add_scatter(
                                        x=[min_val, max_val], 
                                        y=[min_val, max_val], 
                                        mode='lines',
                                        name='Perfect Prediction',
                                        line=dict(dash='dash', color='red')
                                    )
                                    
                                    st.plotly_chart(fig_pred, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not create prediction plot: {e}")

        # --- Optimized Visualization Section ---
        st.header("📈 Interactive Visualizations")
        
        # Initialize chart configs if empty
        if not st.session_state.chart_configs:
            st.info("Add charts using the sidebar controls")
            # Auto-add one chart for quick start
            st.session_state.chart_configs.append({
                "chart_type": suggested_charts[0] if suggested_charts else "Scatter Plot", 
                "id": 0
            })
        
        # Create tabs for charts
        if st.session_state.chart_configs:
            tab_names = [f"📊 Chart {i + 1}" for i in range(len(st.session_state.chart_configs))]
            tabs = st.tabs(tab_names)
            
            figs = []  # For export
            
            for idx, tab in enumerate(tabs):
                with tab:
                    if idx < len(st.session_state.chart_configs):
                        config = st.session_state.chart_configs[idx]
                        
                        # Chart configuration
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            chart_type = st.selectbox(
                                "Chart Type",
                                CHART_OPTIONS,
                                index=CHART_OPTIONS.index(config["chart_type"]) if config["chart_type"] in CHART_OPTIONS else 0,
                                key=f"chart_type_{idx}"
                            )
                            st.session_state.chart_configs[idx]["chart_type"] = chart_type
                        
                        with col2:
                            # Smart column suggestions based on chart type
                            if chart_type in ["Scatter Plot", "Line Chart"]:
                                x_options = numeric_cols + datetime_cols
                                y_options = numeric_cols
                            elif chart_type == "Bar Chart":
                                x_options = categorical_cols
                                y_options = numeric_cols
                            elif chart_type == "Histogram":
                                x_options = numeric_cols
                                y_options = []
                            elif chart_type in ["Box Plot", "Violin Plot"]:
                                x_options = categorical_cols + [None]
                                y_options = numeric_cols
                            elif chart_type == "Pie Chart":
                                x_options = categorical_cols
                                y_options = numeric_cols
                            else:
                                x_options = list(df_filtered.columns)
                                y_options = numeric_cols
                            
                            x_axis = st.selectbox(
                                "X-axis",
                                x_options if x_options else ["No suitable columns"],
                                key=f"x_axis_{idx}"
                            ) if x_options else None
                        
                        with col3:
                            y_axis = st.selectbox(
                                "Y-axis", 
                                y_options if y_options else ["No suitable columns"],
                                key=f"y_axis_{idx}"
                            ) if y_options else None
                        
                        # Advanced options
                        with st.expander("🎛️ Chart Options"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                color_col = st.selectbox(
                                    "Color by",
                                    [None] + categorical_cols,
                                    key=f"color_{idx}"
                                ) if categorical_cols else None
                                
                                theme = st.selectbox(
                                    "Theme",
                                    THEME_OPTIONS,
                                    key=f"theme_{idx}"
                                )
                            
                            with col_b:
                                custom_title = st.text_input(
                                    "Title",
                                    f"{chart_type} - {selected_file}",
                                    key=f"title_{idx}"
                                )
                                
                                height = st.slider(
                                    "Height (px)",
                                    300, 800, 500,
                                    key=f"height_{idx}"
                                )
                        
                        # Generate chart
                        if chart_type == "Correlation Heatmap":
                            if len(numeric_cols) >= 2:
                                try:
                                    # Sample for performance
                                    plot_df = df_filtered.sample(n=min(5000, len(df_filtered)), random_state=42)
                                    corr_matrix = plot_df[numeric_cols].corr()
                                    
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                                    sns.heatmap(
                                        corr_matrix, mask=mask, annot=True, 
                                        cmap='viridis', fmt='.2f', 
                                        linewidths=0.5, ax=ax
                                    )
                                    ax.set_title(custom_title)
                                    st.pyplot(fig, use_container_width=True)
                                    figs.append(fig)
                                except Exception as e:
                                    st.error(f"Correlation heatmap failed: {e}")
                            else:
                                st.warning("Need at least 2 numeric columns")
                        
                        elif chart_type == "Map View":
                            if lat_col and lon_col and lat_col in df_filtered.columns and lon_col in df_filtered.columns:
                                try:
                                    map_data = df_filtered[[lat_col, lon_col]].dropna()
                                    if len(map_data) > 1000:
                                        map_data = map_data.sample(n=1000, random_state=42)
                                        st.info("⚡ Showing 1K points for map performance")
                                    
                                    if not map_data.empty:
                                        st.map(map_data, height=height)
                                        st.success(f"📍 Displayed {len(map_data)} locations")
                                    else:
                                        st.warning("No valid coordinates found")
                                except Exception as e:
                                    st.error(f"Map visualization failed: {e}")
                            else:
                                st.warning("Map requires latitude/longitude columns")
                        
                        else:
                            # Standard plotly charts with caching
                            if x_axis or y_axis:
                                try:
                                    # Create chart hash for caching
                                    chart_params = f"{chart_type}_{x_axis}_{y_axis}_{color_col}_{theme}"
                                    chart_hash = str(hash(chart_params + df_hash))
                                    
                                    fig = create_plotly_chart(
                                        chart_type, chart_hash, x_axis, y_axis,
                                        color_col, custom_title, theme
                                    )
                                    
                                    if fig:
                                        fig.update_layout(height=height)
                                        st.plotly_chart(fig, use_container_width=True)
                                        figs.append(fig)
                                    else:
                                        st.error("Failed to create chart")
                                        
                                except Exception as e:
                                    st.error(f"Chart creation failed: {e}")
                            else:
                                st.info("Please select chart axes above")
                        
                        # Export options for each chart
                        st.markdown("---")
                        export_col1, export_col2, export_col3 = st.columns(3)
                        
                        with export_col1:
                            if st.button(f"📥 Data CSV", key=f"csv_{idx}"):
                                csv_data = convert_df_to_csv(df_filtered)
                                st.download_button(
                                    "Download CSV",
                                    csv_data,
                                    f"chart_{idx}_data.csv",
                                    "text/csv",
                                    key=f"csv_download_{idx}"
                                )
                        
                        with export_col2:
                            if len(figs) > idx and figs[idx] and hasattr(figs[idx], 'to_json'):
                                if st.button(f"📊 Chart JSON", key=f"json_{idx}"):
                                    try:
                                        chart_json = figs[idx].to_json()
                                        st.download_button(
                                            "Download JSON",
                                            chart_json,
                                            f"chart_{idx}.json",
                                            "application/json",
                                            key=f"json_download_{idx}"
                                        )
                                    except Exception as e:
                                        st.error(f"JSON export failed: {e}")
                        
                        with export_col3:
                            if len(figs) > idx and figs[idx]:
                                if st.button(f"🖼️ PNG Image", key=f"png_{idx}"):
                                    try:
                                        if hasattr(figs[idx], 'to_image'):
                                            img_bytes = figs[idx].to_image(format="png", width=1200, height=800)
                                        elif hasattr(figs[idx], 'savefig'):
                                            buf = io.BytesIO()
                                            figs[idx].savefig(buf, format="png", bbox_inches="tight", dpi=150)
                                            img_bytes = buf.getvalue()
                                        else:
                                            img_bytes = None
                                        
                                        if img_bytes:
                                            st.download_button(
                                                "Download PNG",
                                                img_bytes,
                                                f"chart_{idx}.png",
                                                "image/png",
                                                key=f"png_download_{idx}"
                                            )
                                    except Exception as e:
                                        st.error(f"PNG export failed: {e}")

        # --- Performance Optimized Data Preview ---
        with st.expander("📋 Data Preview & Info"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Overview**")
                overview_data = {
                    "Metric": ["Total Rows", "Columns", "Memory Usage", "File Size"],
                    "Value": [
                        f"{len(df):,}",
                        f"{len(df.columns)}",
                        f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                        f"{df.size * 8 / 1024**2:.1f} MB"  # Rough estimate
                    ]
                }
                st.dataframe(pd.DataFrame(overview_data), hide_index=True)
            
            with col2:
                st.markdown("**Column Types**")
                type_data = {
                    "Type": ["Numeric", "Categorical", "DateTime", "Other"],
                    "Count": [
                        len(numeric_cols),
                        len(categorical_cols), 
                        len(datetime_cols),
                        len(df.columns) - len(numeric_cols) - len(categorical_cols) - len(datetime_cols)
                    ]
                }
                st.dataframe(pd.DataFrame(type_data), hide_index=True)
            
            # Show sample data (limited for performance)
            st.markdown("**Sample Data (First 100 rows)**")
            sample_size = min(100, len(df_filtered))
            display_df = df_filtered.head(sample_size)
            
            # Limit columns for display performance
            if len(display_df.columns) > 20:
                display_df = display_df.iloc[:, :20]
                st.info("⚡ Showing first 20 columns for performance")
            
            st.dataframe(display_df, use_container_width=True, height=300)

        # --- Advanced Export & Collaboration ---
        st.header("💾 Export & Collaboration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Session Management")
            
            if st.button("💾 Save Session", key="save_session_btn"):
                try:
                    session_data = {
                        "chart_configs": st.session_state.chart_configs,
                        "filter_state": st.session_state.filter_state,
                        "performance_metrics": st.session_state.performance_metrics,
                        "timestamp": datetime.now().isoformat(),
                        "dataset": selected_file,
                        "dataset_shape": df.shape,
                        "model_count": len(st.session_state.trained_models)
                    }
                    session_json = json.dumps(session_data, indent=2, default=str)
                    
                    st.download_button(
                        "📥 Download Session",
                        session_json,
                        f"session_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        key="session_download_btn"
                    )
                    st.success("✅ Session saved!")
                except Exception as e:
                    st.error(f"Session save failed: {e}")
            
            # Load session
            uploaded_session = st.file_uploader("📤 Load Session", type="json", key="session_upload")
            if uploaded_session:
                try:
                    session_data = json.load(uploaded_session)
                    st.session_state.chart_configs = session_data.get("chart_configs", [])
                    st.session_state.filter_state = session_data.get("filter_state", {})
                    
                    st.success(f"✅ Session loaded from {session_data.get('timestamp', 'unknown date')}")
                    st.info("Refresh the page to apply all settings")
                    
                    # Show session info
                    st.json({
                        "Dataset": session_data.get("dataset", "Unknown"),
                        "Charts": len(session_data.get("chart_configs", [])),
                        "Filters": len(session_data.get("filter_state", {})),
                        "Models": session_data.get("model_count", 0)
                    })
                    
                except Exception as e:
                    st.error(f"Session load failed: {e}")
        
        with col2:
            st.subheader("📊 Dashboard Export")
            
            if st.button("🚀 Generate Dashboard", key="generate_dashboard_btn"):
                try:
                    with st.spinner("Generating optimized dashboard..."):
                        # Gather results for export
                        cluster_result = None
                        rf_result = None
                        
                        # Check for cached ML results
                        for key, result in st.session_state.trained_models.items():
                            if key.startswith("kmeans_") and cluster_result is None:
                                cluster_result = result
                            elif key.startswith("rf_") and rf_result is None:
                                rf_result = result
                        
                        html_content = generate_dashboard_html(
                            df_filtered, eda_data, cluster_result, rf_result, figs
                        )
                        
                        st.download_button(
                            "📥 Download Dashboard HTML",
                            html_content,
                            f"dashboard_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            "text/html",
                            key="dashboard_download_btn"
                        )
                        st.success("✅ Dashboard generated successfully!")
                        
                        # Show dashboard stats
                        dashboard_stats = {
                            "Charts": len(figs),
                            "Data Points": len(df_filtered),
                            "ML Models": len([k for k in st.session_state.trained_models.keys()]),
                            "File Size": f"{len(html_content) / 1024:.1f} KB"
                        }
                        st.json(dashboard_stats)
                        
                except Exception as e:
                    st.error(f"Dashboard generation failed: {e}")

        # --- Performance Summary ---
        if st.session_state.trained_models or st.session_state.chart_configs:
            with st.expander("⚡ Performance Summary"):
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    st.metric("🎯 Cached Models", len(st.session_state.trained_models))
                
                with perf_col2:
                    app_runtime = time.time() - st.session_state.app_start_time
                    st.metric("⏱️ Session Time", f"{app_runtime:.1f}s")
                
                with perf_col3:
                    st.metric("📊 Active Charts", len(st.session_state.chart_configs))
                
                # Performance tips
                st.markdown("""
                **⚡ Performance Tips:**
                - **Smart Sampling**: Large datasets auto-sampled for speed
                - **Caching**: Models and computations cached between interactions  
                - **Lazy Loading**: Heavy operations triggered only when needed
                - **Memory Optimization**: Efficient data structures and processing
                """)

        # --- Help Section ---
        with st.sidebar:
            with st.expander("❓ Help & Tips"):
                st.markdown("""
                **🚀 Quick Start Guide:**
                1. **Upload Data**: Drag & drop your files
                2. **Explore**: Check the EDA summary
                3. **Filter**: Use smart filters to focus
                4. **Visualize**: Create interactive charts
                5. **Model**: Try ML clustering/prediction
                6. **Export**: Download results & dashboards

                **⚡ Performance Features:**
                - **Auto-sampling** for large datasets
                - **Smart caching** for repeated operations
                - **Optimized filtering** with DuckDB
                - **Session persistence** for collaboration

                **🔧 Troubleshooting:**
                - **Slow performance?** → Try filtering large datasets first
                - **Chart errors?** → Check column selections
                - **ML failures?** → Enable preprocessing option
                - **Memory issues?** → Use "Clear Cache" button
                
                **💡 Pro Tips:**
                - Use demo dataset for quick testing
                - Save sessions for team collaboration
                - Export dashboards for presentations
                - Monitor performance metrics in sidebar
                """)

    except Exception as e:
        st.error(f"Application error: {e}")
        logging.error(f"Main application error: {e}")
        
        # Recovery options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload App", key="reload_btn"):
                st.rerun()
        with col2:
            if st.button("🗑️ Reset Everything", key="reset_btn"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical application error: {e}")
        logging.critical(f"Critical error in main: {e}")
        
        st.markdown("""
        ### 🚨 Application Error Recovery
        
        **Try these steps:**
        1. **Refresh** the page in your browser
        2. **Clear browser cache** if issues persist  
        3. **Check file formats** - ensure CSV, Excel, or JSON
        4. **Reduce file size** if working with very large datasets
        
        **Performance Notes:**
        - This app uses advanced caching for optimal speed
        - Large datasets (>100K rows) are automatically sampled
        - ML models are cached between interactions
        - Session state persists your work
        """)
        
        if st.button("🔄 Emergency Reload"):
            st.rerun()