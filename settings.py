import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR_KAGGLE = os.path.join(BASE_DIR, "kaggle-competitions")

DEBUG = True

PROJECT_NAME = 'Kaggle'
LOG_DIR = os.path.join(BASE_DIR, 'kaggle-log')
DATA_DIR = os.path.join(BASE_DIR, 'kaggle-data')