# Eve Analytics Developer Guide

This guide is for developers extending or contributing to **Eve Analytics**, a zero-cost data analytics platform. It covers code structure, key functions, optimizations, and contribution guidelines. Built solo in 10 days on no budget and power outages, the project emphasizes lightweight, maintainable code.

For user-focused instructions, see `docs/user_guide.md`. For function references, see `docs/api_reference.md`.

## Code Structure

The project consists of three main Python files, each representing a version, with shared helper functions for modularity. Code is organized to minimize memory usage and support low-spec hardware.

- **app.py(fastest)**: The standard version with basic data loading, filtering, EDA, K-Means clustering, and simple visualizations.
  - Key Sections: Data loading (`load_data`), filtering (`apply_duckdb_filters`), EDA (`compute_eda_summary`), ML (`perform_kmeans_optimized`), visualizations (`create_plotly_chart`).
  
- **app.py(plus)**: Builds on Eve with enhanced ML (Random Forest), additional charts, and performance features.
  - Key Additions: Random Forest training (`perform_random_forest_optimized`), advanced metrics (`enhanced_regression_metrics`, `enhanced_classification_metrics`), enhanced preprocessing (`preprocess_rf_cached`).

- **app.py(ultra)**: The enterprise-grade version with neural networks (MLP), AutoML, NLP, anomaly detection, and advanced UI.
  - Key Additions: MLP training (`train_mlp_model`), AutoML comparison (`generate_ml_model_comparison`), NLP (`process_advanced_natural_query`), anomaly detection (`detect_anomalies_enterprise`), feature engineering (`perform_automated_feature_engineering`), advanced EDA (`generate_advanced_eda_report`).

- **Shared Modules/Functions**:
  - Data Handling: `load_data_enterprise` (Eve Ultra), `calculate_data_quality_score`.
  - Filtering: `apply_duckdb_filters`.
  - Visualizations: `create_plotly_chart`, `generate_enterprise_dashboard_html`.
  - ML Helpers: `evaluate_model_enhanced`, `preprocess_kmeans_cached`.
  - Utilities: `safe_sql_identifier`, `find_lat_lon_columns`, `convert_df_to_csv`.
  - Constants: File types, chart options, ML models, etc., defined at the top for easy modification.

The code uses Streamlit for the UI, Pandas for data manipulation, Scikit-learn for ML, Plotly/Seaborn for visualizations, and DuckDB for efficient querying. Functions are decorated with `@st.cache_data` or `@st.cache_resource` for performance, with limits (e.g., max_entries=5) to control memory.

## Key Optimizations
Eve Analytics is designed for low-spec hardware (4GB RAM, tested on HP Probook 6560b):
- **Smart Sampling**: Automatically samples large datasets (e.g., 15K rows for ML training, 10K for clustering) to prevent memory overflow.
  - Example: In `perform_kmeans_optimized`, sample if `len(data) > 20000`.
- **Caching Strategy**: Uses Streamlit's caching with limits to avoid recomputation while controlling memory.
  - Example: `@st.cache_data(max_entries=5)` for data loading and preprocessing.
- **Hardware-Aware Processing**: Adaptive batch sizes (e.g., MiniBatchKMeans for large data) and early stopping in models (e.g., MLP with `early_stopping=True`).
- **Memory Efficiency**: Downcasts data types (e.g., float64 to float32), uses sparse structures for categorical encoding, and limits filter counts (e.g., 4 categorical filters).
- **Session State Management**: Stores models and configs in `st.session_state` for persistence, with JSON backups for power outages.
- **Offline Compatibility**: No internet required after setup; all dependencies are local.

These optimizations allow complex features like NLP and AutoML on minimal hardware.

## Contributing
Contributions are welcome, especially those improving low-resource performance or adding features like new ML algorithms (e.g., XGBoost for AutoML).
1. **Fork and Clone**:
   ```bash
   git clone https://github.com/yourusername/eve-analytics.git
   cd eve-analytics
   ```
2. **Create a Branch**:
   ```bash
   git checkout -b feature-name
   ```
3. **Develop**:
   - Write modular functions with type hints (e.g., `def func(param: str) -> Dict[str, Any]:`).
   - Add logging: `logging.info("Feature executed")` for debugging.
   - Test on 4GB RAM: Use small datasets and monitor memory with `memory_profiler` (optional pip install).
   - Ensure compatibility: Avoid heavy libraries; stick to existing dependencies.
4. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Add new ML algorithm"
   git push origin feature-name
   ```
5. **Submit a Pull Request**:
   - On GitHub, create a PR describing changes, tested hardware, and memory impact.
   - Include updates to documentation (e.g., add to `docs/api_reference.md`).

Follow Python PEP 8 style; use Markdown for comments in complex functions.

## Development Tips
- **Low-Resource Workflow**: Use VS Code with minimal extensions (<200MB RAM) or Notepad++. Test features in isolation (e.g., run ML functions separately).
- **Memory Profiling**: Decorate functions with `@profile` from `memory_profiler`:
  ```python
  from memory_profiler import profile
  @profile
  def train_mlp_model(...):
      # Your code
  ```
- **Power Outage Recovery**: Implement session backups:
  ```python
  def save_session_state():
      with open("session_backup.json", "w") as f:
          json.dump(st.session_state.to_dict(), f)
  ```
  Call this periodically or on button press.
- **Testing**: Use the demo dataset for quick tests. Run `pytest` (free) for unit tests if added.
- **Debugging**: Enable logging in `app.log` and check for errors. Use `st.error` for user-facing messages.
- **Adding Features**: Prioritize lightweight additions (e.g., new anomaly methods). Test on your HP Probook to ensure <4GB RAM compatibility.

## Resources
- **API Reference**: See `docs/api_reference.md` for function details.
- **FAQ & Troubleshooting**: `docs/faq_troubleshooting.md` for common issues.
- **Tutorial**: `docs/tutorial.md` for hands-on examples.
- **External**: Streamlit docs (streamlit.io), Scikit-learn user guide (scikit-learn.org).
