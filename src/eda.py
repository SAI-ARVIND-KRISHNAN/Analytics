import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import PLOT_DIR

def run_eda(df):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Top 10 usage
    top_apps = df.sort_values(by="total_time", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='total_time', y='app_name', data=top_apps, palette='rocket')
    plt.title("Top 10 Most Used Apps")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/top_10_apps.png")
    plt.close()

    # Category distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(y='category', data=df, order=df['category'].value_counts().index)
    plt.title("App Category Distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/category_dist.png")
    plt.close()
