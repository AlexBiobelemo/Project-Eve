Advanced Data Visualization Platform
Interactive data analysis and visualization, optimized for low-spec hardware
Overview
The Advanced Data Visualization Platform is a Streamlit-based web application for data ingestion, exploratory data analysis (EDA), interactive visualizations, and machine learning (ML). Built solo in 8 days, it runs efficiently on low-spec hardware (4GB RAM, second-gen Intel Core i5, Intel HD Graphics 3000), leveraging smart caching, sampling, and DuckDB for performance. With 2025 Streamlit advancements, it rivals enterprise tools like Tableau, offering dynamic layouts, data cleaning, neural networks, and conversational queries.
Features
Data Ingestion

Supported Formats: CSV, Excel, JSON (batch uploads).
Performance: Auto-samples large datasets (>100K rows), handles encoding errors (UTF-8, Latin-1), and caches data via @st.cache_data.
Error Handling: Robust fallback for malformed files; logs issues to app.log.

Exploratory Data Analysis (EDA)

Metrics: Displays dataset shape, memory usage, and column types (numeric, categorical, datetime).
Insights: Auto-detects high correlations, missing data, duplicates, and mixed types.
Interactive Cleaning: st.data_editor (Streamlit 2025) for in-place editing, with buttons for removing duplicates, imputing NaNs (mean/median/mode/drop), clipping outliers (Z-score), and type conversion.
Auto-Cleaning: Suggests fixes (e.g., drop high-missing columns, convert types) on sampled data (10K rows), cached for speed.

Smart Filtering

Engine: DuckDB for fast SQL-based filtering on categorical, numeric, and datetime columns.
Optimization: Limits filters (3 categorical, 5 numeric) and samples large datasets (500K rows) to ensure <20s response on low-spec hardware.
UI: Streamlit 2025 st.chat_input for conversational queries (e.g., “Clean NaNs in sales”), parsed via regex to trigger filters or cleaning.

Interactive Visualizations

Chart Types: Scatter, Line, Bar, Histogram, Box, Violin, Pie, Correlation Heatmap, Map View (auto-detects lat/lon).
Dynamic Layouts: Streamlit 2025 flex containers for responsive dashboards (e.g., side-by-side charts, centered metrics).
Sparklines: Inline trend visuals (Streamlit 2025 st.experimental_sparklines) for quick insights.
Customization: Themes (Plotly, Seaborn, High Contrast), color-by columns, adjustable heights.
Performance: Samples 10K rows for charts, caches via @st.cache_data.

Machine Learning

K-Means Clustering: Auto-suggests optimal k using silhouette scores, with MiniBatchKMeans for large datasets. Supports preprocessing and outlier removal (IsolationForest).
Random Forest: Regression/classification with feature selection, hyperparameter tuning, and caching (@st.cache_resource).
Neural Networks: Scikit-learn MLPClassifier/MLPRegressor (2 layers, 50-100 neurons) for lightweight deep learning, training in ~10-20s on 50K rows.
Anomaly Detection: IsolationForest and Z-score methods, visualized in scatter plots.
Performance: Samples 50K rows, limits tuning iterations (6), and caches models.

Export & Collaboration

Outputs: CSV, JSON, PNG for data/charts; HTML dashboards with EDA, ML, and visuals.
Session Management: Save/load session state (charts, filters, metrics) as JSON.
Accessibility: ARIA labels on widgets, high-contrast themes for WCAG compliance.

Performance Optimizations

Caching: Uses @st.cache_data and @st.cache_resource for data, charts, and models.
Sampling: Auto-samples large datasets (10K-50K rows) for EDA, visualizations, and ML.
Lightweight: Optimized for 4GB RAM, with <20s training times and no crashes on 100K+ rows.
Monitoring: Displays load times, memory usage, and cached model counts in sidebar.

Requirements

Python: 3.8+
Libraries: streamlit, pandas, plotly, seaborn, matplotlib, openpyxl, duckdb, numpy, scikit-learn, joblib, pickle
Hardware: Runs on 4GB RAM, second-gen Intel Core i5, Intel HD Graphics 3000.

Installation

Clone the repository:git clone https://github.com/your-username/advanced-data-viz-platform.git
cd advanced-data-viz-platform


Install dependencies:pip install -r requirements.txt


Run the app:streamlit run app.py



Usage

Launch: Run streamlit run app.py and access via localhost:8501.
Upload Data: Drag-and-drop CSV, Excel, or JSON files (batch supported).
Explore:
View EDA metrics and insights in the “Exploratory Data Analysis” section.
Use the “Data Cleaning” expander to edit data, remove duplicates, or impute NaNs.
Query data conversationally via the “AI Assistant” tab (e.g., “Show sales by region”).


Filter: Apply smart filters in the sidebar (categorical, numeric, datetime).
Visualize: Add charts via sidebar, configure with flex containers, and view sparklines.
Model: Run K-Means, Random Forest, MLP, or anomaly detection in the “Machine Learning” section.
Export: Download data, charts, or dashboards; save sessions for collaboration.
Monitor: Check performance metrics (load times, memory, cached models) in the sidebar.

Screenshots

EDA Dashboard: 
Interactive Chart: 
ML Results: 

Performance Notes

Large Datasets: Auto-sampled to 10K-50K rows for speed.
Caching: Data, charts, and models cached to minimize recomputation.
Low-Spec Hardware: Optimized for 4GB RAM, with training times ~10-20s.
Troubleshooting:
Slow performance? Apply filters first or clear cache.
Chart errors? Verify column selections.
ML failures? Enable preprocessing in ML settings.



Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/AmazingFeature).
Commit changes (git commit -m 'Add AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.

License
MIT License. See LICENSE for details.
Contact
For issues or feedback, open a GitHub issue or contact [your-email@example.com].
