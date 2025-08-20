# Technical Specification for Advanced Data Visualization Platform

## Overview
The Advanced Data Visualization Platform is a web-based application built with Streamlit for interactive data analysis, visualization, and machine learning. It supports multiple file formats, advanced filtering, and ML algorithms like K-Means and Random Forest.

## System Architecture
- **Frontend**: Streamlit (Python-based, rendered as HTML/CSS/JavaScript)
- **Backend**: Python 3.8+, DuckDB for querying, scikit-learn for ML
- **Data Storage**: In-memory Pandas DataFrames, with optional DuckDB for large datasets
- **Deployment**: Local server or cloud platforms (Streamlit Cloud, Heroku, AWS)

## Dependencies
- **Core Libraries**:
  - `streamlit>=1.20.0`: Web interface
  - `pandas>=1.5.0`: Data manipulation
  - `numpy>=1.23.0`: Numerical computations
  - `duckdb>=0.8.0`: Fast querying
- **Visualization**:
  - `plotly>=5.10.0`: Interactive charts
  - `seaborn>=0.11.0`: Heatmaps
  - `matplotlib>=3.5.0`: Static plots
- **Machine Learning**:
  - `scikit-learn>=1.2.0`: K-Means, Random Forest, preprocessing
- **File Handling**:
  - `openpyxl>=3.0.0`: Excel support
- **Logging**:
  - `logging`: Error tracking to `app.log`

## Features
- **Data Ingestion**: Supports CSV, Excel, JSON files with batch upload
- **EDA**: Statistical summaries, data types, missing values, correlation insights
- **Data Cleaning**: Editable DataFrame, missing value handling (drop, mean, median)
- **Filtering**: Advanced filters using DuckDB (categorical, numeric, datetime, regex)
- **Visualization**:
  - Chart Types: Scatter, Line, Bar, Histogram, Box, Violin, Pie, Heatmap, Map
  - Customization: Themes, color palettes, font sizes, legend positions
- **Machine Learning**:
  - K-Means Clustering: Auto-k suggestion, preprocessing, outlier removal, metrics (Inertia, Silhouette Score, Davies-Bouldin Index, Elbow Method)
  - Random Forest: Regression/classification, hyperparameter tuning, feature importance, metrics (MSE, R2, Accuracy, Precision, Recall, F1, CV Mean, CV Std, CV Min)
- **Export**: CSV, PNG, JSON for data/charts; HTML for dashboards
- **Collaboration**: Save/load session configurations

## Data Flow
1. **Upload**: User uploads files via Streamlit interface.
2. **Processing**: Files are loaded into Pandas DataFrames.
3. **Filtering**: DuckDB applies filters for performance.
4. **Analysis**: EDA and ML algorithms process data.
5. **Visualization**: Plotly/Seaborn render charts.
6. **Export**: Results are downloadable as files or HTML.

## Security
- **Data Privacy**: Data is processed in-memory; no external storage unless exported.
- **Input Validation**: File uploads restricted to CSV, Excel, JSON.
- **Logging**: Errors logged to `app.log` for debugging.

## Scalability
- **Data Size**: Handles up to 1M rows efficiently with DuckDB.
- **Caching**: Uses `@st.cache_data` and `@st.cache_resource` for performance.
- **Limitations**: Large datasets (>1M rows) may require sampling or higher memory.

## Extensibility
- **New Features**: Add chart types by extending `CHART_OPTIONS` and `plot_functions`.
- **ML Models**: Integrate additional scikit-learn models in `perform_*` functions.
- **Custom Filters**: Extend `apply_duckdb_filters` for new filter types.

## Minimum Requirements
- **Python**: 3.8+
- **Memory**: 4GB (8GB recommended for large datasets)
- **Disk**: 500MB for dependencies and logs
- **Browser**: Modern browsers with JavaScript enabled

## Future Enhancements
- Support for additional file formats (e.g., Parquet)
- Real-time data streaming
- Advanced ML models (e.g., XGBoost, neural networks)
- API integration for external data sources