from sklearn.cluster import KMeans

def encode_labels(df, n_clusters=4):
    # Cluster apps based on average usage
    app_profiles = df.groupby('app_name')['total_time'].mean().reset_index()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    app_profiles['cluster'] = kmeans.fit_predict(app_profiles[['total_time']])

    # Merge cluster label as pseudo-category
    df = df.merge(app_profiles[['app_name', 'cluster']], on='app_name')
    df['category_encoded'] = df['cluster']

    return df, kmeans
