# Eve Analytics: Zero-Cost, Low-Resource Data Science 🚀


## Introduction

Eve Analytics is a suite of three Streamlit-based data analytics platforms—**Eve**, **Eve Plus**, and **Eve Ultra**—designed for enterprise-grade data analysis. Built by Alex Alagoa Biobelemo in 10 days on a 4GB RAM HP Probook , this project pushes the boundaries of creativity under extreme constraints (zero cost, frequent power outages). From basic visualizations to advanced machine learning and conversational AI, Eve Analytics empowers users in resource-constrained environments.

Key highlights:
- **Zero-Cost Development**: Created with $0 using open-source tools like Streamlit, Pandas, Plotly, and Scikit-learn.
- **Low-Resource Optimized**: Tested on a 4GB RAM system with hardware-aware sampling, caching, and adaptive processing.
- **Inspiration**: A testament to maximizing creativity with minimal resources, overcoming constant electricity downtime.

## Features Comparison

| Feature                  | Eve (Standard)                          | Eve Plus (Enhanced)                     | Eve Ultra (Enterprise-Grade)            |
|--------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| **Data Loading**         | CSV, Excel, JSON with basic handling    |  Multiple encodings, quality checks    |  Batch uploads, quality scoring, metadata |
| **Data Processing**      | Basic DuckDB filtering                  |  Smart sampling (>1M rows)             |  Automated feature engineering, advanced quality analysis |
| **Visualizations**       | Scatter, Line, Bar, Histogram, Box      |  Violin, Pie, Map, Correlation Heatmap |  Anomaly Plot, responsive layouts, accessibility themes |
| **Machine Learning**     | K-Means clustering with basic metrics   |  Random Forest (regression/classification) |  MLP neural networks, AutoML comparison, advanced clustering |
| **Anomaly Detection**    | None                                    | IsolationForest, basic outlier removal  | Multiple methods (IsolationForest, Z-Score, Modified Z-Score, IQR) with visualizations |
| **Natural Language Processing** | None                               | None                                    | Conversational AI for queries (cleaning, filtering, stats, ML) |
| **Performance Optimizations** | Basic caching                      | Advanced caching, session state         | Hardware-aware (4GB RAM), adaptive sampling, recovery options |
| **Export/Reporting**     | Basic CSV, HTML dashboard               |  JSON chart export                     |  Themed HTML dashboards, JSON analytics reports |
| **EDA Capabilities**     | Basic summary and insights              |  Correlation heatmap, chart suggestions |  Comprehensive stats, correlations, outliers |
| **Accessibility**        | Basic Streamlit layout                  | Responsive design                       | High Contrast, Colorblind Friendly themes, responsive flex containers |
| **Other**                | Simple UI                               | Performance dashboard, session management | Conversational AI tab, advanced UI controls |

## Installation

1. **Prerequisites**:
   - Python 3.8 (free download from python.org; tested on low-spec hardware).
   - No internet required after installation (offline compatible).
   - Dependencies (install via pip; total ~200MB disk space):
     ```bash
     pip install streamlit pandas plotly seaborn scikit-learn openpyxl duckdb numpy scipy
     ```
   - On low-RAM systems: Install one package at a time to avoid memory issues.

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/eve-analytics.git
   cd eve-analytics
   ```
   (If no Git, download ZIP from GitHub.)

3. **Run the App**:
   - For Eve Ultra (recommended): `streamlit run eve_ultra.py`
   - For Eve Plus: `streamlit run eve_plus.py`
   - For Eve: `streamlit run eve.py`
   - Browser opens at `localhost:8501`. Use a lightweight browser like Firefox.

*Note*: For large files on 4GB RAM, add `--server.maxUploadSize 50` to the command.

## Quick Start

1. **Launch the App**:
   - Run `streamlit run eve_ultra.py` (or the version of choice).
   
2. **Load Data**:
   - Upload a CSV/Excel/JSON file or click "Load Demo Dataset" (8,000 rows of enterprise-like sales data with realistic missing values and outliers).

3. **Basic Usage**:
   - **Explore**: View EDA summary in the "Analytics" tab.
   - **Visualize**: Create a Scatter Plot in "Visualizations".
   - **Analyze**: Use the AI Assistant in Eve Ultra: "Show stats for sales_amount".
   - **Model**: Train an MLP in "ML Studio" to predict revenue.
   - **Detect Anomalies**: Run IsolationForest in "Anomaly Detection".
   - **Export**: Generate an HTML dashboard or JSON report.

See `docs/tutorial.md` for a detailed walkthrough and `docs/user_guide.md` for version-specific instructions.

## System Requirements
- **Minimum**: 4GB RAM, 1GHz CPU (tested on HP Probook 6560b with power outages).
- **Recommended**: 8GB RAM for larger datasets (>1M rows).
- **Optimizations**: 
  - Smart sampling (15K rows for ML, 10K for clustering).
  - Adaptive processing to handle low-spec hardware.
  - Offline functionality after setup.
- **Dependencies**: Open-source and free; no cloud or internet required for core features.

## Contributing
We welcome contributions, especially optimizations for low-resource systems!
1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feature-name`.
3. Develop and test on 4GB RAM hardware.
4. Add logging for debugging: `logging.info("Your message")`.
5. Commit changes: `git commit -m "Add feature"`.
6. Submit a pull request.

See `docs/developer_guide.md` for code structure and guidelines.

## About the Developer
Built solo by [Your Name] in 10 days with $0, overcoming constant electricity downtime on a 4GB RAM HP Probook 6560b. This project showcases creativity and technical skill in resource-constrained environments, proving enterprise-grade analytics can be achieved with minimal resources.

- **Inspiration**: Pushing boundaries of innovation with so little—zero cost, low-spec hardware, and unreliable power.
- **Contact**: [your.email@example.com] | LinkedIn: [your-linkedin] | GitHub: [your-github]

## License
MIT License - free to use, modify, and distribute. See `LICENSE` file for details.

---

🌟 **Star this repo** to support low-resource data science! If this project inspires you, share your story or contribute. 🚀
