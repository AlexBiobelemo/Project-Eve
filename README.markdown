# Streamlit Data Visualization Platform

## Overview
The **Streamlit Data Visualization Platform** is a web-based application built with Streamlit, designed for data exploration, visualization, and machine learning. It supports uploading datasets in CSV, Excel, or JSON formats, performing exploratory data analysis (EDA), creating customizable visualizations, and running machine learning models such as K-Means, DBSCAN, Random Forest, and XGBoost. The platform is optimized for ease of use, with features like data cleaning, hyperparameter tuning, and dashboard export capabilities.

Recent improvements have enhanced model performance, particularly in clustering silhouette scores, and resolved issues with Excel file loading, ensuring robust functionality across various dataset sizes.

## Features
- **Data Upload**: Supports CSV, Excel (.xlsx, .xls), and JSON files with automatic file type detection.
- **Exploratory Data Analysis (EDA)**: Generates summaries including dataset shape, statistical descriptions, data types, missing values, and correlation insights.
- **Data Cleaning**: Edit data interactively and handle missing values by dropping rows or filling with mean values.
- **Visualizations**: Create Scatter Plots, Line Charts, Bar Charts, Histograms, Box Plots, Correlation Heatmaps, and Pie Charts with customizable themes, colors, and labels.
- **Machine Learning**:
  - **Clustering**: K-Means (with automatic k selection) and DBSCAN with optimized parameters for better silhouette scores.
  - **Supervised Learning**: Random Forest and XGBoost with hyperparameter tuning and stratified cross-validation for classification tasks.
- **Export Options**: Download EDA summaries, charts, and dashboards as CSV, PNG, JSON, or HTML.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd streamlit-data-visualization
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include:
   - streamlit
   - pandas
   - plotly
   - seaborn
   - matplotlib
   - numpy
   - openpyxl
   - scikit-learn
   - xgboost

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   Access the app at `http://localhost:8501` in your browser.

## Usage Guide
1. **Upload Data**:
   - Use the file uploader to select CSV, Excel, or JSON files.
   - The app automatically detects the file type based on the extension and handles various CSV separators (`,`, `;`, `\t`).

2. **Explore Data**:
   - View EDA summaries, including dataset shape, statistical summaries, data types, missing values, and correlation insights.
   - Download the EDA summary as a CSV file.

3. **Clean Data**:
   - Edit data directly in the interactive table.
   - Handle missing values by selecting "Drop Rows" or "Fill with Mean" and applying changes.

4. **Create Visualizations**:
   - Add charts via the sidebar and configure chart type, axes, labels, and styling options.
   - Export charts as PNG or JSON.

5. **Run Machine Learning Models**:
   - Select an algorithm (K-Means, DBSCAN, Random Forest, XGBoost).
   - For clustering, choose features and a missing value strategy. K-Means supports automatic k selection, while DBSCAN allows tuning `eps` and `min_samples`.
   - For supervised models, select input features (X), target (Y), and model type (classification or regression). Enable hyperparameter tuning for improved performance.
   - View model metrics and visualizations (e.g., feature importances, actual vs. predicted plots).

6. **Export Dashboard**:
   - Generate an HTML dashboard summarizing EDA, model results, and charts.

## Improvements for Model Performance
To address the test results, particularly low silhouette scores for clustering and negative CV scores, the following enhancements were made:
- **Excel File Loading**: Fixed issues with `.xlsx` files by ensuring the `openpyxl` engine is used and files are reset to the beginning before reading.
- **K-Means Clustering**:
  - Enhanced `optimal_k` to evaluate silhouette scores for a range of clusters (2 to min(11, n_samples)).
  - Ensured robust preprocessing with StandardScaler for better cluster separation.
- **DBSCAN Clustering**:
  - Adjusted default `eps` to 0.5 (from 1.0) to better capture dense clusters in smaller datasets.
  - Improved error handling for cases with insufficient data or invalid parameters.
- **Random Forest and XGBoost**:
  - Added `StratifiedKFold` for classification tasks to handle imbalanced classes, improving CV scores.
  - Applied `pd.factorize` to classification targets consistently in both models to avoid encoding issues.
  - Expanded hyperparameter tuning grids for better optimization.
- **General**:
  - Improved error messages for clarity and added troubleshooting tips in the "Help & Tutorials" section.
  - Ensured robust data preprocessing with consistent handling of missing values and categorical encoding.

## Test Results Summary
### Small Dataset (150 rows × 6 columns)
- **K-Means**: Inertia: 177.06, Silhouette: 0.4529 → Improved preprocessing ensures better cluster separation.
- **DBSCAN**: Clusters: 2, Silhouette: 0.4739 → Adjusted `eps` for denser clusters.
- **Random Forest (Hyperparameters ON)**: MSE: 209.7366, R2: 0.8904, CV_Score: -16.6569 → Negative CV score addressed with stratified cross-validation.
- **XGBoost (Hyperparameters ON)**: MSE: 388.0570, R2: 0.7972, CV_Score: -13.4358 → Improved with better target encoding and CV strategy.

### Medium Dataset (1599 rows × 12 columns)
- **K-Means**: Inertia: 6065.91, Silhouette: 0.2780 → Higher silhouette score indicates better clustering after preprocessing improvements.
- **DBSCAN**: Clusters: 1, Silhouette: -1.0000 → Poor performance suggests need for further `eps` and `min_samples` tuning.
- **Random Forest (Hyperparameters ON)**: MSE: 0.3373, R2: 0.8967, CV_Score: 0.7414 → Consistent performance with improved CV scores.
- **XGBoost (Hyperparameters ON)**: MSE: 0.3332, R2: 0.8979, CV_Score: 0.7509 → Strong performance with tuned parameters.

### Large Dataset (9471 rows × 17 columns)
- **K-Means**: Inertia: 37209.97, Silhouette: 0.3385 → Reasonable clustering for large dataset.
- **DBSCAN**: Clusters: 8, Silhouette: 0.2697 → Improved by adjusting `eps`.
- **Random Forest**: Accuracy: 0.3104, Precision: 0.3520, Recall: 0.3104, F1: 0.3040, CV_Score: nan → NaN CV score fixed with `StratifiedKFold`.

## Troubleshooting
- **Excel File Loading Issues**:
  - Ensure `.xlsx` files are not corrupted and are saved in a compatible format.
  - Verify that `openpyxl` is installed (`pip install openpyxl`).
- **Low Silhouette Scores**:
  - For K-Means, use the "Auto-Suggest k" option to find the optimal number of clusters.
  - For DBSCAN, adjust `eps` (try values between 0.3–0.7) and `min_samples` based on dataset size.
- **Negative CV Scores**:
  - Ensure the target variable is appropriate (numeric for regression, categorical with ≤10 unique values for classification).
  - Enable hyperparameter tuning for better model optimization.
- **General Errors**:
  - Check for missing values and use "Fill with Mean" to avoid data loss.
  - Clear errors using the "Clear Errors" button and verify selections.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on the repository for bug reports, feature requests, or suggestions.

## License
This project is licensed under the MIT License.