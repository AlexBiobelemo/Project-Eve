# User Guide for Advanced Data Visualization Platform

## Introduction
The Advanced Data Visualization Platform is an intuitive web application for data analysis, visualization, and machine learning. This guide walks you through its features, from uploading data to generating insights and exporting results.

## Getting Started
1. **Access the Platform**:
   - Local: Run `streamlit run app.py` and open `http://localhost:8501`.
   - Cloud: Access the deployed URL (e.g., Streamlit Cloud, Heroku).
2. **Upload Data**:
   - Select file type (CSV, Excel, JSON) from the dropdown.
   - Upload one or more files using the file uploader.
3. **Select Dataset**: Choose a dataset from the uploaded files to analyze.

## Key Features

### 1. Exploratory Data Analysis (EDA)
- **Location**: Main section, under "Exploratory Data Analysis (EDA)".
- **Functionality**:
  - View dataset shape (rows × columns).
  - Expand sections for statistical summaries, data types, missing values, and insights.
  - Download EDA summary as CSV.
- **Usage**: Review summaries to understand data distribution and identify issues (e.g., missing values, correlations).

### 2. Data Cleaning
- **Location**: Main section, under "Data Cleaning".
- **Functionality**:
  - Edit data directly in a table.
  - Handle missing values (drop rows, fill with mean/median).
  - Apply changes to update the dataset.
- **Usage**: Clean data to prepare it for analysis or modeling.

### 3. Data Filtering
- **Location**: Sidebar, under "Data Filtering".
- **Functionality**:
  - Filter categorical columns with multiselect.
  - Filter numeric columns with sliders.
  - Filter datetime columns with date pickers.
  - Apply regex filters for text columns.
- **Usage**: Narrow down data to focus on specific subsets (e.g., sales in 2023).

### 4. Machine Learning
- **Location**: Main section, under "Machine Learning".
- **K-Means Clustering**:
  - Enable auto-k suggestion or set k manually (2–10).
  - Optionally preprocess features or remove outliers.
  - View results: cluster assignments, inertia, Silhouette Score, Davies-Bouldin Index, and Elbow Method plot.
- **Random Forest**:
  - Choose regression or classification.
  - Select input features (X) and target (Y).
  - Optionally preprocess, tune hyperparameters, or remove outliers.
  - View metrics (e.g., MSE, R2, Accuracy, CV Mean, CV Std, CV Min), feature importance, and actual vs. predicted plots.
- **Usage**: Run clustering to group data or Random Forest to predict outcomes.

### 5. Visualizations
- **Location**: Main section, under "Generated Visualizations".
- **Functionality**:
  - Add multiple charts via the sidebar.
  - Select chart type (Scatter, Line, Bar, Histogram, Box, Violin, Pie, Heatmap, Map).
  - Configure axes, labels, themes, colors, and legend positions.
  - View performance metrics (DuckDB vs. Pandas filtering).
  - Export charts as PNG/JSON or data as CSV.
- **Usage**: Create visualizations to explore data patterns and share results.

### 6. Collaboration
- **Location**: Main section, under "Collaboration Features".
- **Functionality**:
  - Save session state (charts, filters) as JSON.
  - Load a saved JSON configuration to restore settings.
- **Usage**: Share configurations with team members for collaborative analysis.

### 7. Dashboard Export
- **Location**: Main section, under "Dashboard Export".
- **Functionality**: Export EDA, ML results, and charts as an HTML file.
- **Usage**: Generate a portable dashboard for presentations or reports.

## Tips
- **Large Datasets**: Enable preprocessing or outlier removal for faster ML.
- **Chart Suggestions**: Check sidebar for AI-suggested chart types based on data.
- **Performance**: DuckDB filtering is faster than Pandas for large datasets.
- **Help**: Expand the "Help & Tutorials" section in the sidebar for guidance.

## Troubleshooting
- **Upload Errors**: Ensure files are in CSV, Excel, or JSON format.
- **Visualization Issues**: Verify axis selections match data types.
- **ML Errors**: Check for sufficient data and compatible feature/target types.

For further assistance, refer to the [Technical Specification](Technical_Specification.md) or contact support.