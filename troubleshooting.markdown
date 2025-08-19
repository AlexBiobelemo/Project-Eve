# Troubleshooting Guide

This guide addresses common issues encountered when using the Streamlit Data Visualization Platform and provides solutions to ensure smooth operation.

## Excel File Loading Issues
- **Problem**: `.xlsx` files fail to load with errors like "Failed to load file: Invalid file format."
- **Solution**:
  - Verify that the file is a valid Excel file (.xlsx or .xls) and not corrupted.
  - Ensure `openpyxl` is installed: `pip install openpyxl`.
  - Check if the file contains data in a tabular format with clear headers.
  - Try saving the file in a different format (e.g., CSV) as a workaround.

## Low Silhouette Scores in Clustering
- **Problem**: K-Means or DBSCAN produces low silhouette scores (e.g., 0.1731 or -1.0000).
- **Solution**:
  - **K-Means**:
    - Enable "Auto-Suggest k" to select the optimal number of clusters based on silhouette scores.
    - Ensure at least two numeric columns are selected and contain varied data (avoid columns with all identical values).
    - Use "Fill with Mean" for missing values to avoid data loss.
  - **DBSCAN**:
    - Adjust `eps` to a lower value (e.g., 0.3–0.7) for denser clusters, especially in smaller datasets.
    - Increase `min_samples` for larger datasets to reduce noise points.
    - Visualize clusters using Scatter Plots to assess if clusters are meaningful.

## Negative or NaN Cross-Validation Scores
- **Problem**: Random Forest or XGBoost shows negative CV scores (e.g., -16.6569) or NaN (e.g., large dataset classification).
- **Solution**:
  - Ensure the target variable (Y) is appropriate:
    - For regression, the target must be numeric.
    - For classification, the target should have ≤10 unique values or be categorical.
  - Enable hyperparameter tuning to optimize model performance.
  - Check for imbalanced classes in classification tasks; the updated code uses `StratifiedKFold` to address this.
  - Verify that sufficient data remains after dropping missing values (at least 20 rows).

## Chart Rendering Failures
- **Problem**: Charts fail to render with errors like "Chart rendering failed: Invalid parameters."
- **Solution**:
  - Ensure required axes are selected (e.g., X and Y for Scatter Plot, X for Histogram).
  - For Bar or Pie Charts, confirm that the aggregation function (mean, sum, count) produces non-empty results.
  - Check for missing or invalid data in selected columns; use the "Data Cleaning" section to address missing values.

## General Errors
- **Problem**: Persistent errors or unexpected behavior.
- **Solution**:
  - Click the "Clear Errors" button to reset error messages and try again.
  - Review the "Help & Tutorials" section in the sidebar for detailed guidance.
  - Check the `app.log` file for detailed error logs if running locally.
  - Ensure all dependencies are installed and up-to-date (`pip install -r requirements.txt`).

## Contact
For further assistance, open an issue on the repository or contact the maintainer.