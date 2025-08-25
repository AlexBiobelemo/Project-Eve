# Eve Analytics API Reference

This document provides a reference for key functions in **Eve Analytics**, with a focus on **Eve Ultra** for its enterprise-grade capabilities. It is intended for developers extending or integrating the codebase. All functions are optimized for low-resource systems (4GB RAM, tested on HP Probook 6560b) and use lightweight libraries like Pandas, Scikit-learn, Plotly, and DuckDB.

For code structure and contribution guidelines, see `docs/developer_guide.md`. For usage examples, see `docs/user_guide.md` and `docs/tutorial.md`.

## Functions

Below are the most critical functions in **Eve Ultra**, with parameters, return types, and examples. Functions from **Eve** and **Eve Plus** are included where they differ significantly.

### `load_data_enterprise`
- **Description**: Loads CSV, Excel, or JSON files with enterprise-grade error handling, data quality scoring, and metadata generation.
- **Parameters**:
  - `file_content` (bytes): File content as bytes.
  - `file_name` (str): Name of the file (e.g., "data.csv").
  - `file_type` (str): File type ("CSV", "Excel", "JSON").
- **Returns**: `pandas.DataFrame` with metadata in `df.attrs` (e.g., `quality_score`, `load_time`), or `None` if loading fails.
- **Example**:
  ```python
  with open("sales_data.csv", "rb") as f:
      df = load_data_enterprise(f.read(), "sales_data.csv", "CSV")
  if df is not None:
      print(df.attrs['quality_score'])  # e.g., 92.5 (0-100 scale)
      print(df.attrs['missing_values'])  # e.g., {'sales_amount': 5}
  ```
- **Notes**: Uses multiple encoding attempts (UTF-8, Latin-1) and calculates quality based on missing values, duplicates, and type consistency. Optimized for <4GB RAM with sampling for large files (>1M rows).

### `apply_duckdb_filters`
- **Description**: Applies categorical and numeric filters to a DataFrame using DuckDB for efficient querying.
- **Parameters**:
  - `df` (pandas.DataFrame): Input dataset.
  - `categorical_filters` (List[Dict]): List of filters (e.g., `{'column': 'region', 'values': ['North America']}`).
  - `numeric_filters` (List[Dict]): List of range filters (e.g., `{'column': 'sales_amount', 'min': 100, 'max': 10000}`).
- **Returns**: `pandas.DataFrame` with filtered data, or original DataFrame if no filters.
- **Example**:
  ```python
  filters = [
      {'column': 'region', 'values': ['North America']},
      {'column': 'sales_amount', 'min': 100, 'max': 5000}
  ]
  filtered_df = apply_duckdb_filters(df, categorical_filters=[filters[0]], numeric_filters=[filters[1]])
  print(len(filtered_df))  # Reduced row count
  ```
- **Notes**: Used in all versions (Eve, Eve Plus, Eve Ultra). Limits to 4 filters for performance on low-spec hardware.

### `process_advanced_natural_query`
- **Description**: Processes natural language queries for data operations (Eve Ultra only), supporting cleaning, filtering, stats, visualization, and ML tasks.
- **Parameters**:
  - `query` (str): User query (e.g., "Clean NaNs in sales_amount" or "Train regression model using sales, profit to predict revenue").
  - `df` (pandas.DataFrame): Input dataset.
- **Returns**: `Dict[str, Any]` with keys: `action` (str, e.g., "stats"), `message` (str, result description), `success` (bool), `data` (optional, e.g., filtered DataFrame or chart).
- **Example**:
  ```python
  response = process_advanced_natural_query("Show stats for sales_amount", df)
  if response['success']:
      print(response['message'])  # e.g., "Mean: 1234.56, Std: 567.89"
  ```
- **Notes**: Uses regex-based parsing with predefined patterns. Optimized for minimal memory overhead. Fallback to keyword matching for robustness.

### `train_mlp_model`
- **Description**: Trains a Multi-Layer Perceptron (MLP) for regression or classification, optimized for low-spec hardware (Eve Ultra only).
- **Parameters**:
  - `df` (pandas.DataFrame): Input dataset.
  - `x_cols` (List[str]): Feature columns (e.g., `['sales_amount', 'profit']`).
  - `y_col` (str): Target column (e.g., `revenue`).
  - `model_type` (str): "regression" or "classification".
  - `hidden_layers` (Tuple[int, ...]): Neuron counts per layer (default: `(50, 50)`).
  - `max_iter` (int): Maximum iterations (default: 200).
- **Returns**: `Dict[str, Any]` with keys: `model` (trained MLP), `metrics` (e.g., R², ROC AUC), `training_time` (float).
- **Example**:
  ```python
  result = train_mlp_model(df, ['sales_amount', 'profit'], 'revenue', 'regression')
  print(result['metrics']['r2'])  # e.g., 0.85
  print(result['training_time'])  # e.g., 2.3 seconds
  ```
- **Notes**: Uses early stopping and samples data to 15K rows for 4GB RAM compatibility. Metrics include R², RMSE for regression; ROC AUC, F1 for classification.

### `perform_random_forest_optimized`
- **Description**: Trains a Random Forest model for regression or classification (Eve Plus, Eve Ultra).
- **Parameters**:
  - `df` (pandas.DataFrame): Input dataset.
  - `x_cols` (List[str]): Feature columns.
  - `y_col` (str): Target column.
  - `model_type` (str): "regression" or "classification".
  - `n_estimators` (int): Number of trees (default: 100).
  - `max_depth` (int or None): Max tree depth (default: None).
- **Returns**: `Dict[str, Any]` with keys: `model` (trained Random Forest), `metrics` (e.g., R², ROC AUC), `feature_importance` (Dict[str, float]).
- **Example**:
  ```python
  result = perform_random_forest_optimized(df, ['sales', 'profit'], 'deal_size', 'classification')
  print(result['metrics']['roc_auc'])  # e.g., 0.92
  print(result['feature_importance'])  # e.g., {'sales': 0.6, 'profit': 0.4}
  ```
- **Notes**: Samples to 15K rows and uses `@st.cache_resource` for performance. Enhanced metrics in Eve Ultra.

### `detect_anomalies_enterprise`
- **Description**: Detects outliers using multiple methods (IsolationForest, Z-Score, Modified Z-Score, IQR) with visualizations (Eve Ultra only).
- **Parameters**:
  - `df` (pandas.DataFrame): Input dataset.
  - `columns` (List[str]): Columns to analyze (e.g., `['sales_amount', 'profit']`).
  - `method` (str): Anomaly detection method (e.g., "IsolationForest", "ZScore").
  - `contamination` (float): Expected outlier proportion for IsolationForest (default: 0.1).
- **Returns**: `Dict[str, Any]` with keys: `anomalies` (DataFrame with anomaly flags), `scores` (anomaly scores), `plot` (Plotly figure).
- **Example**:
  ```python
  result = detect_anomalies_enterprise(df, ['sales_amount', 'profit'], 'IsolationForest', 0.1)
  print(result['anomalies'].head())  # DataFrame with 'is_anomaly' column
  result['plot'].show()  # View anomaly scatter plot
  ```
- **Notes**: Samples to 10K rows for visualization. Supports multiple methods for robust analysis.

### `generate_enterprise_dashboard_html`
- **Description**: Creates a themed HTML dashboard with EDA insights, charts, and ML results (Eve Ultra).
- **Parameters**:
  - `df` (pandas.DataFrame): Input dataset.
  - `charts` (List[plotly.graph_objects.Figure]): List of Plotly charts.
  - `ml_results` (List[Dict]): List of ML model results.
  - `anomaly_results` (Dict): Anomaly detection results.
  - `theme` (str): Theme for styling ("Professional", "Dark", "Colorful").
- **Returns**: `str` (HTML content for the dashboard).
- **Example**:
  ```python
  charts = [create_plotly_chart(df, 'scatter', 'sales_amount', 'profit')]
  ml_results = [train_mlp_model(df, ['sales', 'profit'], 'revenue', 'regression')]
  anomaly_results = detect_anomalies_enterprise(df, ['sales'], 'IsolationForest')
  html_content = generate_enterprise_dashboard_html(df, charts, ml_results, anomaly_results, 'Professional')
  with open("dashboard.html", "w") as f:
      f.write(html_content)
  ```
- **Notes**: Lightweight HTML generation (<1MB output). Includes accessibility options.

### `perform_automated_feature_engineering`
- **Description**: Generates derived features (e.g., polynomial, interactions, bins) for ML tasks (Eve Ultra only).
- **Parameters**:
  - `df` (pandas.DataFrame): Input dataset.
  - `numeric_cols` (List[str]): Numeric columns for engineering.
- **Returns**: `pandas.DataFrame` with new features (e.g., `sales_squared`, `sales_profit_interaction`).
- **Example**:
  ```python
  new_df = perform_automated_feature_engineering(df, ['sales_amount', 'profit'])
  print(new_df.columns)  # e.g., ['sales_amount', 'profit', 'sales_amount_squared']
  ```
- **Notes**: Memory-efficient with sampling and sparse encoding for categoricals.

## Additional Notes
- **Caching**: Most functions use `@st.cache_data` or `@st.cache_resource` with `max_entries=5` to limit memory usage.
- **Error Handling**: Functions include try-except blocks with logging (`logging.error`) and user-friendly messages via `st.error`.
- **Low-Resource Optimization**: All functions sample large datasets and downcast data types (e.g., float32) for 4GB RAM compatibility.
- **Extending Functions**: Add new ML algorithms or anomaly methods in `eve_ultra.py`. Update `docs/api_reference.md` accordingly.

For a full list of functions, review `eve.py`, `eve_plus.py`, and `eve_ultra.py`. Contact [your.email@example.com] for questions.
