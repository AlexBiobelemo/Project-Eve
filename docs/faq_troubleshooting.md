# Eve Analytics FAQ and Troubleshooting

This document addresses frequently asked questions (FAQs) and common issues for **Eve Analytics** (**Eve**, **Eve Plus**, **Eve Ultra**), optimized for low-resource systems (4GB RAM, tested on HP Probook 6560b). It is designed for all users, including non-technical users, data scientists, and developers, with solutions tailored to zero-cost development and power outage challenges.

For usage instructions, see `docs/user_guide.md`. For developers, see `docs/developer_guide.md` and `docs/api_reference.md`.

## Frequently Asked Questions

### **Q: Can Eve Analytics run on 4GB RAM?**
**A**: Yes, all versions are optimized for 4GB RAM (tested on HP Probook 6560b). Key optimizations include:
- Smart sampling (15K rows for ML, 10K for clustering in Eve Ultra).
- Caching with limits (`@st.cache_data`, max_entries=5).
- Downcasting data types (e.g., float64 to float32).
Use the demo dataset (8,000 rows) for best performance on low-spec hardware.

### **Q: How do I handle large datasets (>1M rows)?**
**A**: Eve Ultra automatically samples large datasets (15K rows for ML, 10K for visualizations). To optimize:
- Manually reduce dataset size before uploading (e.g., split CSV files).
- Clear cache via the "Emergency Reset" button in Eve Ultra.
- In `eve_ultra.py`, adjust sampling limits in `load_data_enterprise` if needed:
  ```python
  if len(df) > 1000000:
      df = df.sample(n=15000, random_state=42)
  ```

### **Q: What if I experience power outages during analysis?**
**A**: Save your session state to recover progress:
1. In Eve Ultra, click "Save Session" to export to `session_backup.json`.
2. Or manually save:
   ```python
   with open("session_backup.json", "w") as f:
       json.dump(st.session_state.to_dict(), f)
   ```
3. On restart, load the session via the "Load Session" button or:
   ```python
   with open("session_backup.json", "r") as f:
       st.session_state.update(json.load(f))
   ```
- Save work every 15 minutes to a USB drive or local folder:
  ```bash
  cp *.json backup/
  ```

### **Q: Which file formats are supported?**
**A**: CSV, Excel, and JSON. Eve Ultra supports multiple encodings (UTF-8, Latin-1) and batch uploads. Use the demo dataset (`enterprise_demo.csv`) for testing.

### **Q: How do I use the conversational AI in Eve Ultra?**
**A**: In the "AI Assistant" tab, type plain English queries like:
- "Show stats for sales_amount" (displays mean, std, etc.).
- "Clean NaNs in profit_margin" (imputes missing values).
- "Create a scatter plot of sales_amount vs profit_margin".
- "Train regression model using sales, profit to predict revenue".
See `docs/user_guide.md` for more examples.

### **Q: Can I use Eve Analytics offline?**
**A**: Yes, all core features work offline after installing dependencies. The demo dataset and cached models ensure functionality without internet.

### **Q: How do I export results?**
**A**: In Eve Ultra:
- Export filtered data as CSV from the "Data" tab.
- Generate an HTML dashboard (with themes) or JSON report from the "Export" button in the "Analytics" or "ML Studio" tabs.
- Example:
  ```python
  html_content = generate_enterprise_dashboard_html(df, charts, ml_results, anomaly_results, "Professional")
  with open("dashboard.html", "w") as f:
      f.write(html_content)
  ```

## Troubleshooting

### **Memory Error**
- **Symptoms**: App crashes with "MemoryError" or slows significantly.
- **Solutions**:
  - Reduce dataset size (<100K rows) or use Eve Ultra’s smart sampling.
  - Clear cache: Click "Emergency Reset" in Eve Ultra or run:
    ```python
    st.cache_data.clear()
    st.cache_resource.clear()
    ```
  - Downcast data types manually:
    ```python
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')
    ```
  - Close other apps (e.g., browser tabs) to free RAM.

### **File Loading Failure**
- **Symptoms**: Error like "Failed to load file" or "Invalid format".
- **Solutions**:
  - Ensure file is CSV, Excel, or JSON. Check encoding (UTF-8, Latin-1).
  - In Eve Ultra, use `load_data_enterprise` with multiple encoding attempts:
    ```python
    df = load_data_enterprise(file_content, "data.csv", "CSV")
    ```
  - Verify file integrity (e.g., no corrupted rows). Use the demo dataset for testing.
  - Check file size: Keep <50MB for 4GB RAM compatibility.

### **Slow Performance**
- **Symptoms**: App lags or takes long to process.
- **Solutions**:
  - Use a lightweight browser (e.g., Firefox with minimal tabs).
  - Limit filters to 4 categorical or numeric in the sidebar.
  - Enable "Auto preprocessing" in ML tabs to optimize data.
  - For ML tasks, reduce features or use smaller datasets (e.g., 1K rows for testing).
  - Monitor performance in Eve Ultra’s "Performance Metrics" section.

### **Chart Rendering Issues**
- **Symptoms**: Visualization fails to display or shows incorrect data.
- **Solutions**:
  - Ensure valid X/Y axes (e.g., numeric for Scatter Plot).
  - Reduce plot complexity: Set smaller height/width in "Visualizations" tab.
  - Clear Plotly cache:
    ```python
    st.cache_data.clear()
    ```
  - Switch to simpler charts (e.g., Bar instead of Violin) for large datasets.

### **Model Training Failure**
- **Symptoms**: Error like "Invalid input" or model fails to train.
- **Solutions**:
  - Check for missing values in features or target. Use "Data Cleaning" tab in Eve Ultra.
  - Reduce feature count (e.g., 2-3 columns) for faster training.
  - Ensure correct model type (regression/classification) in "ML Studio".
  - Test with the demo dataset and minimal features:
    ```python
    result = train_mlp_model(df, ['sales_amount'], 'profit_margin', 'regression')
    ```

### **Conversational AI Misinterpretation**
- **Symptoms**: AI Assistant returns incorrect results or "Query not understood".
- **Solutions**:
  - Use clear, specific queries (e.g., "Filter sales_amount > 1000" instead of "Show high sales").
  - Check supported commands in `docs/user_guide.md`.
  - Update `process_advanced_natural_query` patterns if adding new query types:
    ```python
    patterns.append({"pattern": r"filter\s+(\w+)\s*>\s*(\d+)", "action": "filter_numeric"})
    ```

### **Power Outage Recovery**
- **Symptoms**: Lost progress due to sudden shutdown.
- **Solutions**:
  - Save session state regularly (see FAQ above).
  - Backup filtered data as CSV:
    ```python
    df.to_csv("backup_data.csv", index=False)
    ```
  - Store code changes in Git:
    ```bash
    git add .
    git commit -m "Backup progress"
    ```

## Additional Tips
- **Low-Resource Systems**: Use Eve for small datasets (<10K rows), Eve Plus for medium datasets (<1M rows), and Eve Ultra for advanced features with sampling.
- **Logging**: Check `app.log` for errors (enabled in all versions).
- **Testing**: Use the demo dataset (`enterprise_demo.csv`) to troubleshoot without external files.
- **Community**: Report issues or suggestions at [your.email@example.com] or GitHub Issues.

For hands-on examples, see `docs/tutorial.md`. Happy troubleshooting! 🚀
