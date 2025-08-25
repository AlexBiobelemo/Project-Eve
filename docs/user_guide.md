# Eve Analytics User Guide

This guide provides step-by-step instructions for using **Eve**, **Eve Plus**, and **Eve Ultra** for data analysis. It is designed for both technical users (e.g., data scientists) and non-technical users (e.g., business analysts or students), with a focus on simplicity and low-resource optimization. All versions run efficiently on 4GB RAM systems, and examples use the built-in demo dataset (8,000 rows of sales data with realistic missing values and outliers).

For installation and system requirements, see `README.md`. For developers extending the code, see `docs/developer_guide.md`.

## Getting Started with Any Version

1. **Launch the App**:
   - Open a terminal and navigate to the project folder.
   - Run the desired version:
     - Eve: `streamlit run eve.py`
     - Eve Plus: `streamlit run eve_plus.py`
     - Eve Ultra: `streamlit run eve_ultra.py` (recommended for advanced features)
   - The app opens in your browser at `localhost:8501`. Use a lightweight browser like Firefox to minimize RAM usage.

2. **Load Data**:
   - In the "Upload Files" section, select your file type (CSV, Excel, JSON) and upload one or more files.
   - Alternatively, click "Load Demo Dataset" to use the sample `enterprise_demo.csv` (columns include `sales_amount`, `profit_margin`, `customer_segment`, `region`, etc.).
   - Tip: For large files on 4GB RAM, start with <100K rows. Eve Ultra automatically samples large datasets.

3. **Select Dataset**:
   - Choose the uploaded file from the dropdown menu.

4. **Apply Global Filters (Sidebar)**:
   - Use the "Smart Data Filtering" expander to filter categorical (e.g., select "North America" for `region`) or numeric columns (e.g., range slider for `sales_amount`).
   - Filters apply across all tabs; monitor the row reduction info for performance.

## Using Eve (Standard Version)
- **Best For**: Basic exploratory data analysis (EDA) and simple visualizations on small datasets (<10K rows).
- **Key Tabs**:
  - **Exploratory Data Analysis**: View summary statistics, insights, and data quality checks.
  - **Machine Learning**: Run K-Means clustering on numeric columns.
  - **Interactive Visualizations**: Create charts like Scatter Plot or Histogram.

- **Example: Create a Basic Visualization and Clustering**
  1. Load the demo dataset.
  2. In the "Exploratory Data Analysis" section, review metrics (rows, columns) and insights (e.g., "High missing data in profit_margin").
  3. Go to "Interactive Visualizations" and add a chart.
  4. Select "Scatter Plot", X-axis: `sales_amount`, Y-axis: `profit_margin`, Color by: `region`.
  5. In "Machine Learning", select features (`sales_amount`, `profit_margin`), set clusters to 3, and run K-Means.
  6. Export filtered data as CSV from the chart export options.

- **Non-Technical Tip**: Use the demo dataset to explore without your own data. Focus on the visualization tab for quick insights.

## Using Eve Plus (Plus Version)
- **Best For**: Enhanced visualizations and predictive modeling on medium datasets (~100K-1M rows).
- **Key Improvements Over Eve**:
  - Additional charts (Violin Plot, Pie Chart, Map View, Correlation Heatmap).
  - Random Forest models for regression/classification with advanced metrics (R², RMSE, ROC AUC).
  - Performance dashboard and session management.

- **Example: Advanced Visualization and Modeling**
  1. Load the demo dataset.
  2. In "Exploratory Data Analysis", view the correlation heatmap for numeric columns (e.g., correlation between `sales_amount` and `profit_margin`).
  3. Go to "Interactive Visualizations" and create a "Violin Plot" of `sales_amount` by `customer_segment`.
  4. In "Machine Learning", select "Random Forest", features (`sales_amount`, `customer_satisfaction`), target (`profit_margin`), enable "Hyperparameter tuning", and train the model.
  5. Review metrics (e.g., R² > 0.8 indicates good fit) and feature importance chart.
  6. Save the session for later and export an HTML dashboard with insights and charts.

- **Non-Technical Tip**: Use the "Auto-suggest k" for clustering and "Auto preprocessing" for modeling to simplify tasks.

## Using Eve Ultra (Ultra Version)
- **Best For**: Enterprise-grade analytics on large datasets (>1M rows) with AI assistance, neural networks, and anomaly detection.
- **Key Improvements Over Eve Plus**:
  - Tabs for AI Assistant, Analytics, Data Cleaning, Anomaly Detection, Visualizations, and ML Studio.
  - Conversational AI for natural queries (e.g., "Train regression model using sales to predict revenue").
  - MLP neural networks, AutoML comparison, and automated feature engineering.
  - Multiple anomaly detection methods with visualizations.
  - Accessibility themes (High Contrast, Colorblind Friendly) and responsive UI.

- **Example: Full Enterprise Workflow**
  1. Load the demo dataset.
  2. **AI Assistant Tab**: Type "Show stats for sales_amount" (displays mean, std, etc.) or "Clean NaNs in profit_margin" (fills missing values).
  3. **Analytics Tab**: Review advanced EDA report with skewness, kurtosis, and high correlations.
  4. **Data Cleaning Tab**: Click "Analyze Data Quality" for suggestions (e.g., "Impute missing in profit_margin"), then apply them.
  5. **Anomaly Detection Tab**: Select "IsolationForest", features (`sales_amount`, `profit_margin`), set contamination to 0.1, and detect anomalies. View the plot with red points for outliers.
  6. **Visualizations Tab**: Create an "Anomaly Plot" colored by anomaly scores.
  7. **ML Studio Tab**: Select "AutoML Comparison", "regression" task, features (`sales_amount`, `customer_satisfaction`), target (`profit_margin`), and train. Compare Random Forest vs. MLP metrics.
  8. Export a themed HTML dashboard or JSON report with all results.

- **Non-Technical Tip**: Rely on the AI Assistant for most tasks—type plain English queries like "Filter sales_amount > 1000" or "Create a pie chart of region". Switch to "High Contrast" theme for better visibility.

## Advanced Features in Eve Ultra
- **Conversational AI**: Supports data cleaning, stats, filtering, charting, and ML. Examples:
  - "Clean NaNs in sales_amount" → Fills missing values.
  - "Train classification model using sales_amount, profit_margin for target deal_size" → Runs AutoML.
  - "Generate EDA report" → Produces detailed statistics.
- **Automated Feature Engineering**: In ML workflows, automatically creates squared/log features and interactions.
- **Accessibility**: Select themes in the top-right corner for better usability.
- **Exports**: HTML dashboards (themed, with insights) and JSON reports (dataset summary, ML metrics, anomalies).

## Tips for Low-Resource Systems
- **Memory Management**: Use small datasets initially. Clear cache via "Emergency Reset" in Eve Ultra if errors occur.
- **Power Outages**: Save sessions to JSON for recovery: Use the "Save Session" button.
- **Performance**: Enable "Auto preprocessing" in ML tabs to handle mixed data types automatically.
- **Offline Use**: All core features work offline after setup. Use the demo dataset for testing without internet.

## Next Steps
- Practice with the tutorial in `docs/tutorial.md`.
- Check `docs/faq_troubleshooting.md` for solutions to common issues.
- For developers: See `docs/developer_guide.md` and `docs/api_reference.md`.
