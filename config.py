# config.py
import os

# Define the base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the base directory
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'diabetes.csv')
REPORT_HTML_PATH = os.path.join(BASE_DIR, 'reports', 'diabetes_data_profile.html')
REPORT_JSON_PATH = os.path.join(BASE_DIR, 'reports', 'data_quality_report.json')