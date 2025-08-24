# Enterprise Data Analytics Platform

## Overview
The **Enterprise Data Analytics Platform** is a state-of-the-art, AI-powered web application built with Streamlit, designed for advanced data analysis, visualization, and machine learning. Optimized for enterprise use, it supports large-scale datasets, low-resource environments, and provides a highly interactive, user-friendly interface. Key features include natural language query processing, automated data cleaning, anomaly detection, advanced machine learning, and responsive visualizations, all tailored for scalability and performance.

## Features
- **AI-Powered Analytics**: Process natural language queries to clean data, generate statistics, filter datasets, and create visualizations.
- **Data Cleaning Studio**: Automated data quality analysis with intelligent cleaning suggestions (e.g., handling missing values, outliers, and type conversions).
- **Anomaly Detection**: Supports multiple methods (Isolation Forest, Z-Score, Modified Z-Score, IQR) for identifying outliers in numeric data.
- **Machine Learning Studio**: Train advanced models (Random Forest, MLP, AutoML comparisons) with optimized preprocessing and hyperparameter tuning.
- **Interactive Visualizations**: Create dynamic charts (scatter, line, bar, histogram, box, violin, pie, heatmap, map) with customizable themes and accessibility options.
- **Performance Optimization**: Smart sampling, caching, and hardware-aware processing for efficient operation on low-spec systems (4GB RAM supported).
- **Enterprise Reporting**: Generate HTML dashboards and JSON reports for comprehensive data insights.
- **Accessibility**: Supports multiple themes (Light, Dark, High Contrast, Colorblind Friendly) for inclusive user experience.

## Tech Stack
- **Framework**: Streamlit (2025 features for responsive layouts and enhanced UX)
- **Data Processing**: Pandas, NumPy, DuckDB
- **Machine Learning**: Scikit-learn (KMeans, RandomForest, MLP, IsolationForest)
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Additional Libraries**: Openpyxl, Joblib, Scipy
- **Logging**: Python `logging` for enterprise-grade error tracking
- **File Support**: CSV, Excel, JSON

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/enterprise-data-analytics.git
   cd enterprise-data-analytics
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Requirements File** (`requirements.txt`):
   ```text
   streamlit>=1.38.0
   pandas>=2.0.0
   numpy>=1.25.0
   plotly>=5.15.0
   seaborn>=0.13.0
   matplotlib>=3.8.0
   openpyxl>=3.1.0
   duckdb>=0.10.0
   scikit-learn>=1.3.0
   joblib>=1.3.0
   scipy>=1.11.0
   ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Launch the App**: Open your browser and navigate to `http://localhost:8501`.
2. **Upload Data**: Select a file type (CSV, Excel, JSON) and upload your dataset(s). Batch uploads are supported.
3. **Select Dataset**: Choose a dataset from the uploaded files for analysis.
4. **Explore Tabs**:
   - **AI Assistant**: Use natural language queries (e.g., "Clean NaNs in sales", "Show stats for age", "Create bar chart of revenue").
   - **Analytics**: View key insights, trends, and dataset summaries.
   - **Data Cleaning**: Apply automated cleaning suggestions for missing values, outliers, and more.
   - **Anomaly Detection**: Configure and run anomaly detection with customizable parameters.
   - **Visualizations**: Build interactive charts with advanced customization options.
   - **ML Studio**: Train and compare machine learning models with automated preprocessing.
5. **Export Results**: Download HTML dashboards, JSON reports, or processed datasets.

## Example Queries
- **Data Cleaning**: "Clean NaNs in profit_margin"
- **Statistics**: "Show summary of sales_amount"
- **Filtering**: "Filter region equals North America"
- **Charts**: "Create scatter plot of sales_amount vs profit_margin"
- **ML**: "Train classification model using sales_amount, profit_margin to predict customer_segment"
- **Clustering**: "Perform clustering on sales_amount, customer_satisfaction"

## Performance Optimization
- **Smart Sampling**: Automatically samples large datasets (>20,000 rows) for faster processing.
- **Caching**: Utilizes Streamlit's `@st.cache_data` for efficient data loading, EDA, and model training.
- **Low-Resource Support**: Optimized for systems with as little as 4GB RAM, with sub-20s model training times.
- **Error Handling**: Comprehensive logging to `enterprise_app.log` for troubleshooting.

## Accessibility
- **Themes**: Light, Dark, High Contrast, and Colorblind Friendly themes.
- **Responsive Design**: Flex containers and wide layouts for optimal viewing on any device.
- **Zoom & Interaction**: Charts support zoom and interactive controls for enhanced usability.

## Logging
- Logs are saved to `enterprise_app.log` with timestamps, error levels, and detailed messages.
- Example log entry:
  ```
  2025-08-24 23:03:45 - INFO - Data loaded successfully: enterprise_demo.csv
  ```

## Limitations
- Maximum file size depends on available memory (recommended <500MB for low-spec systems).
- Map visualizations require latitude/longitude columns.
- Some advanced features (e.g., AutoML) may require sufficient numeric/categorical columns.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For support or inquiries, contact the development team at support@enterprise-analytics.com.