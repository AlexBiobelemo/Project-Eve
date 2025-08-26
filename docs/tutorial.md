# Eve Analytics Tutorial: Analyzing Sales Data

This tutorial demonstrates how to use **Eve Ultra**, the enterprise-grade version of **Eve Analytics**, to analyze the built-in demo dataset (`enterprise_demo.csv`, 8,000 rows of sales data with columns like `sales_amount`, `profit_margin`, `customer_segment`, `region`, etc.). It covers data loading, exploration, filtering, visualization, machine learning, anomaly detection, and exporting results. The steps are designed to work efficiently on a 4GB RAM system (tested on HP Probook 6560b) and are suitable for technical and non-technical users.

For general usage, see `docs/user_guide.md`. For developers, see `docs/developer_guide.md` and `docs/api_reference.md`. For troubleshooting, see `docs/faq_troubleshooting.md`.

## Prerequisites
- **Eve Analytics** installed (see `README.md` for setup).
- Python 3.8+ and dependencies: `streamlit`, `pandas`, `plotly`, `seaborn`, `scikit-learn`, `openpyxl`, `duckdb`, `numpy`, `scipy`.
- Lightweight browser (e.g., Firefox) to minimize RAM usage.
- No internet required after setup; demo dataset included.

## Step 1: Launch Eve Ultra
1. Open a terminal and navigate to the project folder:
   ```bash
   cd eve-analytics
   ```
2. Run Eve Ultra:
   ```bash
   streamlit run eve_ultra.py
   ```
3. Open `localhost:8501` in a browser. Use Firefox with minimal tabs to save RAM.

## Step 2: Load the Demo Dataset
1. In the "Upload Files" section, click **"Load Demo Dataset"** to use `enterprise_demo.csv` (8,000 rows, ~1MB).
   - Columns include `sales_amount` (numeric), `profit_margin` (numeric), `customer_segment` (categorical), `region` (categorical), `deal_size` (categorical), etc.
2. Select the dataset from the dropdown menu in the sidebar.
   - Note: The dataset has realistic missing values and outliers for testing data cleaning and anomaly detection.

## Step 3: Explore Data
1. Go to the **"Analytics"** tab.
2. Review the **EDA Report**:
   - Dataset metrics: Rows (8,000), columns, memory usage.
   - Insights: Missing data (e.g., "5% missing in profit_margin"), numeric/categorical column counts.
   - Advanced stats: Skewness, kurtosis, high correlations (e.g., `sales_amount` vs. `profit_margin`).
3. Check the **Data Quality Score** (0-100) based on missing values, duplicates, and type consistency.
   - Example: A score of 92 indicates high-quality data with minor issues.

## Step 4: Filter Data
1. In the sidebar, expand **"Smart Data Filtering"**.
2. Apply a categorical filter:
   - Select `region` and choose "North America".
3. Apply a numeric filter:
   - Select `sales_amount` and set range to 100–10,000 using the slider.
4. Check the row count reduction (e.g., from 8,000 to ~2,000 rows) in the sidebar.
   - Tip: Limit to 4 filters to maintain performance on 4GB RAM.

## Step 5: Create Visualizations
1. Go to the **"Visualizations"** tab.
2. Click **"Add Chart"** and select **"Violin Plot"**.
3. Configure:
   - X-axis: `customer_segment` (e.g., "Enterprise", "SMB").
   - Y-axis: `sales_amount`.
   - Color by: `deal_size` (e.g., "Small", "Large").
   - Theme: Select "High Contrast" for accessibility.
4. Click **"Generate Chart"** to view the plot.
   - The violin plot shows the distribution of `sales_amount` across customer segments, colored by deal size.
5. Optionally, create a **Correlation Heatmap**:
   - Select numeric columns (`sales_amount`, `profit_margin`, `customer_satisfaction`).
   - Generate to see correlations (e.g., `sales_amount` and `profit_margin` may have a correlation of 0.75).

## Step 6: Train a Model
1. Go to the **"ML Studio"** tab.
2. Select **"AutoML Comparison"** and task type **"regression"**.
3. Configure:
   - Features: `sales_amount`, `customer_satisfaction`, `deal_size`.
   - Target: `profit_margin`.
   - Enable "Auto preprocessing" (handles missing values, encoding).
   - Enable "Hyperparameter tuning" for better model performance.
4. Click **"Train Advanced Model"**.
5. Review results:
   - Compare Random Forest vs. MLP metrics (e.g., R²: 0.85 for Random Forest, 0.82 for MLP).
   - View feature importance for Random Forest (e.g., `sales_amount`: 60% importance).
   - Check training time (e.g., ~2 seconds on 4GB RAM due to 15K row sampling).
6. Save the trained model to `st.session_state` for reuse.

## Step 7: Detect Anomalies
1. Go to the **"Anomaly Detection"** tab.
2. Select **"IsolationForest"** as the method.
3. Configure:
   - Features: `sales_amount`, `profit_margin`.
   - Contamination: 0.1 (assumes 10% of data are outliers).
4. Click **"Detect Anomalies"**.
5. Review:
   - **Anomaly Plot**: Scatter plot with outliers in red.
   - **Results**: DataFrame with `is_anomaly` column (True/False) and anomaly scores.
   - Example: ~800 rows flagged as outliers (10% of 8,000).
6. Try **"Z-Score"** method for comparison; note differences in flagged outliers.

## Step 8: Use the Conversational AI
1. Go to the **"AI Assistant"** tab.
2. Type queries in plain English:
   - "Show stats for sales_amount" → Displays mean, std, min, max, etc.
   - "Clean NaNs in profit_margin" → Imputes missing values with median.
   - "Filter sales_amount > 1000" → Applies filter and updates dataset.
   - "Create a pie chart of region" → Generates a pie chart.
   - "Train classification model using sales_amount, profit_margin for deal_size" → Runs AutoML and shows metrics.
3. Check the response message for results or errors (e.g., "Query not understood" for invalid inputs).

## Step 9: Export Results
1. Go to the **"Analytics"** or **"ML Studio"** tab.
2. Click **"Export"** and select:
   - **HTML Dashboard**: Choose "Professional" theme. Includes EDA insights, charts, ML results, and anomaly plots.
   - **JSON Report**: Contains dataset summary, ML metrics, and anomaly results.
3. Save the output:
   - HTML: Open `dashboard.html` in a browser (lightweight, <1MB).
   - JSON: Review `report.json` for structured data.
4. Optionally, export filtered data as CSV:
   ```python
   df.to_csv("filtered_data.csv", index=False)
   ```

## Tips for Low-Resource Systems
- **Memory Management**: If errors occur, clear cache via **"Emergency Reset"** or:
  ```python
  st.cache_data.clear()
  st.cache_resource.clear()
  ```
- **Power Outages**: Save session state to recover progress:
  ```python
  with open("session_backup.json", "w") as f:
      json.dump(st.session_state.to_dict(), f)
  ```
  Load on restart: **"Load Session"** button or manual JSON import.
- **Performance**: Use small datasets (<10K rows) for testing. Enable "Auto preprocessing" to minimize manual steps.
- **Backup**: Save work to a USB drive:
  ```bash
  cp *.csv *.json *.html backup/
  ```

## Next Steps
- Explore more features in `docs/user_guide.md` (e.g., advanced NLP queries, accessibility themes).
- Troubleshoot issues in `docs/faq_troubleshooting.md`.
- For developers: Extend functionality using `docs/developer_guide.md` and `docs/api_reference.md`.
- Share feedback or issues at https://www.linkedin.com/in/alex-alagoa-biobelemo/.
