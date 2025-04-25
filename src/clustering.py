from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import PLOT_DIR


def run_clustering(df):
    os.makedirs(PLOT_DIR, exist_ok=True)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['total_time']])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='total_time', y='category', hue='cluster', palette='viridis')
    plt.title("KMeans Clustering")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/clusters.png")
    plt.close()

    return df
