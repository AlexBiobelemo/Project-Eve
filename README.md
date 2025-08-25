# Eve: An AI-Powered Data Analytics Platform

Eve is an open-source, AI-driven data analytics platform designed to turn raw data into actionable insights with maximum efficiency. From automated data cleaning and advanced machine learning to a conversational AI assistant, Eve provides an end-to-end analytics workflow in a single, intuitive interface.

This project was a solo development challenge, built from scratch in just 10 days on a 4GB RAM laptop (HP Probook 6560b) with a $0 budget, often navigating constant power outages. It was an experiment in pushing the boundaries of creativity and performance engineering with minimal resources.


## Key Features (Eve Ultra)
1. **Conversational AI Assistant**: Ask questions and perform actions using natural language. (e.g., "Clean NaNs in sales" or "Create a scatter plot of profit vs sales").
2. **Automated Data Cleaning Studio**: Get intelligent suggestions for handling missing data, converting data types, and managing outliers with one-click fixes.
3. **Anomaly Detection Engine**: Use powerful algorithms like Isolation Forest and Z-Score to automatically identify unusual data points.
4. **Advanced ML Studio**: Train, compare, and analyze multiple models, including Random Forest and Neural Networks (MLP).
5. **AutoML Capabilities**: Automatically compare different models to find the best performer for your specific regression or classification task.
6. **Interactive Visualizations**: Build a wide range of dynamic charts and dashboards with a simple, intuitive UI.
7. **Performance-Obsessed Architecture**: Heavily optimized with smart sampling, advanced caching, and efficient libraries (DuckDB) to run smoothly on low-resource systems.


## Version Showcase
The project is available in three distinct versions, showcasing its evolution:

1. **Eve (Standard)**: The foundational, high-performance data visualization tool.
2. **Eve plus (Advanced)**: Adds a comprehensive suite of advanced metrics for in-depth machine learning model evaluation.
3. **Eve Ultra (Enterprise)**: The complete AI-powered platform with a conversational assistant, AutoML, anomaly detection, and automated data cleaning.


## Quickstart
Get the Eve Ultra platform running on your local machine in a few simple steps:

- Clone the repository:
  git clone https://github.com/your-username/eve-analytics-platform.git

- Install dependencies:
  pip install -r requirements.txt

- Run the application:
  streamlit run Eve_Ultra/app.py

For detailed setup, see the Installation Guide

## Tech Stack
- **Backend**: Python
- **Frontend/UI**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Data Filtering**: DuckDB
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn

### Full Documentation
For a deep dive into the project's architecture, features, and technical implementation, please visit the full documentation in the docs/ directory.

### License
This project is licensed under the MIT License. See the LICENSE file for details.
