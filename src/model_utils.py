from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import joblib
import matplotlib.pyplot as plt
from config import MODEL_DIR, REPORT_DIR, PLOT_DIR


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def train_and_save_models(X_train, X_test, y_train, y_test, label_encoder=None):
    ensure_dirs()

    models = {
        "logreg": LogisticRegression(),
        "tree": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        if len(set(y_train)) < 2:
            print(f"Skipping {name}: Only one class present in y_train.")
            continue

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # If encoder is passed, use class names. Otherwise, use numeric cluster labels.
        class_names = label_encoder.classes_ if label_encoder else sorted(set(y_train))
        report = classification_report(y_test, preds, target_names=[str(cls) for cls in class_names])

        with open(f"{REPORT_DIR}/{name}_report.txt", "w") as f:
            f.write(report)

        joblib.dump(model, f"{MODEL_DIR}/{name}_model.pkl")

        if name == "tree":
            plt.figure(figsize=(12, 6))
            plot_tree(model, feature_names=['total_time'], class_names=[str(cls) for cls in class_names], filled=True)
            plt.savefig(f"{PLOT_DIR}/tree_structure.png")
            plt.close()
