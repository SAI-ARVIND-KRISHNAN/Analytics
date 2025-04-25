import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime

# Config
DATA_PATH = "data"
ARTIFACT_DIR = "artifacts"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_DIR = os.path.join(ARTIFACT_DIR, f"models_v{TIMESTAMP}")
REPORT_DIR = os.path.join(ARTIFACT_DIR, f"reports_v{TIMESTAMP}")
PLOT_DIR = os.path.join(ARTIFACT_DIR, f"plots_v{TIMESTAMP}")

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

def parse_duration(text):
    if pd.isna(text):
        return None
    try:
        parts = text.lower().replace('h', 'h ').replace('m', 'm ').replace('s', 's ').split()
        total_seconds = 0
        for part in parts:
            if 'h' in part:
                total_seconds += int(part.replace('h', '')) * 3600
            elif 'm' in part:
                total_seconds += int(part.replace('m', '')) * 60
            elif 's' in part:
                total_seconds += int(part.replace('s', ''))
        return total_seconds / 60
    except:
        return None

def load_data(directory):
    all_files = [f for f in os.listdir(directory) if f.endswith(".xls")]
    combined_df = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(directory, file)
        print(f"📄 Reading: {file_path}")
        try:
            xls_sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            print(f"❌ Failed to read {file_path}: {e}")
            continue

        sheet_names = list(xls_sheets.keys())
        if not sheet_names:
            continue

        first_sheet = xls_sheets[sheet_names[0]]

        try:
            df = first_sheet.iloc[1:-2]
            df = df.rename(columns={df.columns[0]: 'app_name', df.columns[-1]: 'total_time'})
            df = df[['app_name', 'total_time']]
            df['total_time'] = df['total_time'].apply(parse_duration)
            df.dropna(subset=['total_time'], inplace=True)
            df['app_name'] = df['app_name'].str.strip().str.lower()
            df['user_id'] = os.path.splitext(file)[0]
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Skipped {file_path}: structure mismatch → {e}")

    return combined_df

def run_eda(df):
    top_apps = df.sort_values(by="total_time", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='total_time', y='app_name', data=top_apps, palette='rocket')
    plt.title("Top 10 Most Used Apps")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/top_apps.png")
    plt.close()

def encode_clusters(df, n_clusters=4):
    app_profiles = df.groupby('app_name')['total_time'].mean().reset_index()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    app_profiles['cluster'] = kmeans.fit_predict(app_profiles[['total_time']])
    df = df.merge(app_profiles[['app_name', 'cluster']], on='app_name', how='left')
    df['category_encoded'] = df['cluster']
    return df

def train_and_save_models(X_train, X_test, y_train, y_test):
    models = {
        "logreg": LogisticRegression(),
        "tree": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        if len(set(y_train)) < 2:
            print(f"⚠️ Skipping {name}: Only one class present.")
            continue

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        report = classification_report(y_test, preds)
        with open(f"{REPORT_DIR}/{name}_report.txt", "w") as f:
            f.write(report)

        joblib.dump(model, f"{MODEL_DIR}/{name}_model.pkl")

        if name == "tree":
            plt.figure(figsize=(12, 6))
            plot_tree(model, feature_names=['total_time'], class_names=[str(i) for i in set(y_train)], filled=True)
            plt.savefig(f"{PLOT_DIR}/tree_structure.png")
            plt.close()

def main():
    ensure_dirs()
    df = load_data(DATA_PATH)
    run_eda(df)
    df = encode_clusters(df)

    X = df[['total_time']]
    y = df['category_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_and_save_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
