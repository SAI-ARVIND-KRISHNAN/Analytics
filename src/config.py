import os
from datetime import datetime

# ENV paths
DATA_PATH = os.getenv("DATA_PATH", "data/")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts/")

# Timestamp for versioning
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_DIR = os.path.join(ARTIFACT_DIR, f"models_v{TIMESTAMP}")
REPORT_DIR = os.path.join(ARTIFACT_DIR, f"reports_v{TIMESTAMP}")
PLOT_DIR = os.path.join(ARTIFACT_DIR, f"plots_v{TIMESTAMP}")
