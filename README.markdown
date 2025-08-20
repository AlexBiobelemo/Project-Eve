# Advanced Data Visualization Platform

## Overview

The Advanced Data Visualization Platform is a powerful, Streamlit-based web application for data analysis, visualization, and machine learning. It supports multiple file formats (CSV, Excel, JSON), advanced filtering with DuckDB, interactive visualizations with Plotly, and machine learning with scikit-learn (K-Means, Random Forest). Users can perform exploratory data analysis (EDA), clean data, apply ML models, and export results as interactive dashboards.

## Features

- **Data Ingestion**: Batch upload for CSV, Excel, and JSON files.
- **EDA**: Statistical summaries, data types, missing values, and correlation insights.
- **Data Cleaning**: Edit data, handle missing values (drop, mean, median).
- **Filtering**: Advanced filters (categorical, numeric, datetime, regex) using DuckDB.
- **Visualizations**: Scatter, Line, Bar, Histogram, Box, Violin, Pie, Heatmap, and Map charts with customizable themes and export options.
- **Machine Learning**:
  - K-Means Clustering with auto-k suggestion, Silhouette Score, Davies-Bouldin Index, and Elbow Method.
  - Random Forest (regression/classification) with hyperparameter tuning, feature importance, and metrics (MSE, R2, Accuracy, CV Mean, CV Std, CV Min).
- **Collaboration**: Save/load session configurations as JSON.
- **Export**: Download data as CSV, charts as PNG/JSON, and dashboards as HTML.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AlexBiobelemo/Project-Eve
cd ProjectEve
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

Access at `http://localhost:8501`.

## Usage

1. Upload datasets (CSV, Excel, JSON).
2. Explore data with EDA tools and clean as needed.
3. Apply filters in the sidebar to refine data.
4. Run K-Means or Random Forest models under "Machine Learning".
5. Create and customize visualizations under "Generated Visualizations".
6. Save configurations or export dashboards for sharing.

## Requirements

- Python 3.8+
- Dependencies: `streamlit`, `pandas`, `plotly`, `seaborn`, `matplotlib`, `numpy`, `duckdb`, `scikit-learn`, `openpyxl`
- Browser: Chrome, Firefox, or Safari

## Deployment

See Deployment Guide for local and cloud deployment instructions.

## Documentation

- User Guide: Step-by-step usage instructions.
- Technical Specification: System architecture and details.
- Performance Report: Performance metrics and analysis.

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the repository.

## License

MIT License

## Contact

For support, contact the development team or open an issue on GitHub.