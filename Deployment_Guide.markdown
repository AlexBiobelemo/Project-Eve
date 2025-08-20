# Deployment Guide for Advanced Data Visualization Platform

## Overview

This guide provides detailed instructions for deploying the Advanced Data Visualization Platform, a Streamlit-based application designed for data analysis, visualization, and machine learning. The deployment process covers setting up the environment, installing dependencies, and running the application on a local or cloud-based server.

## Prerequisites

Before deploying the application, ensure the following requirements are met:

- **Operating System**: Linux, macOS, or Windows
- **Python Version**: Python 3.8 or higher
- **Hardware Requirements**:
  - Minimum: 4GB RAM, 2 CPU cores
  - Recommended: 8GB RAM, 4 CPU cores for large datasets
- **Internet Access**: Required for downloading dependencies
- **Web Browser**: Chrome, Firefox, or Safari for accessing the Streamlit interface
- **Git**: For cloning the repository (optional)

## Installation Steps

### 1. Clone the Repository

Clone the project repository from your version control system:

```bash
git clone https://github.com/AlexBiobelemo/Project-Eve
cd ProjectEve
```

### 2. Set Up a Virtual Environment

Create and activate a Python virtual environment to isolate dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```
streamlit>=1.20.0
pandas>=1.5.0
plotly>=5.10.0
seaborn>=0.11.0
matplotlib>=3.5.0
numpy>=1.23.0
openpyxl>=3.0.0
duckdb>=0.8.0
scikit-learn>=1.2.0
```

### 4. Configure Environment Variables

Create a `.env` file in the project root to configure optional settings:

```plaintext
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0
```

- `STREAMLIT_PORT`: Port for the Streamlit server (default: 8501)
- `STREAMLIT_HOST`: Host address (default: 0.0.0.0 for local access)

### 5. Run the Application Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501` in your web browser.

### 6. Cloud Deployment (Optional)

To deploy on a cloud platform (e.g., Streamlit Cloud, Heroku, or AWS), follow these steps:

#### Streamlit Cloud

1. Log in to Streamlit Cloud.
2. Create a new app and link it to your GitHub repository.
3. Specify the main script (`app.py`) and Python version (3.8+).
4. Ensure `requirements.txt` is in the repository root.
5. Deploy the app and access the provided URL.

#### Heroku

1. Install the Heroku CLI and log in:

```bash
heroku login
```

2. Create a Heroku app:

```bash
heroku create my-data-viz-app
```

3. Add a `Procfile` in the project root:

```plaintext
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

4. Deploy to Heroku:

```bash
git push heroku main
```

5. Open the app:

```bash
heroku open
```

### 7. Verify Deployment

- Access the application URL (local or cloud).
- Upload a sample dataset (CSV, Excel, or JSON) to confirm functionality.
- Test visualization and machine learning features (e.g., K-Means clustering, Random Forest).

## Troubleshooting

- **Port Conflict**: If port 8501 is in use, change `STREAMLIT_PORT` in the `.env` file.
- **Dependency Errors**: Ensure all packages in `requirements.txt` are compatible with your Python version.
- **Memory Issues**: For large datasets, increase server memory or enable data sampling in the app.

## Post-Deployment

- **Monitoring**: Check `app.log` for runtime errors.
- **Updates**: Pull the latest code and redeploy for updates.
- **Scaling**: For high traffic, consider cloud platforms with auto-scaling (e.g., AWS EC2, Google Cloud Run).

For further assistance, contact the development team or refer to the User Guide.