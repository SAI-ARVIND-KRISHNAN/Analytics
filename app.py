import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import seaborn as sns
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')


class AppUsageAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.usage_data = None
        self.features = None
        self.models = {}
        self.user_data = {}  # Store per-user data
        self.user_metrics = {}  # Store per-user metrics
        self.comparison_metrics = {}  # Store comparison metrics

    def load_data(self):
        """Load all Excel files from directory and combine data"""
        all_data = []

        for file in os.listdir(self.data_dir):
            if file.endswith('.xls') or file.endswith('.xlsx'):
                file_path = os.path.join(self.data_dir, file)
                try:
                    # Extract user ID from filename (assuming format like "user_123.xlsx")
                    user_id = self._extract_user_id(file)

                    # Load each sheet (assuming consistent format)
                    usage_time = pd.read_excel(file_path, sheet_name=0)
                    usage_count = pd.read_excel(file_path, sheet_name=1)
                    unlocks = pd.read_excel(file_path, sheet_name=2)

                    # Extract date
                    date = usage_time.iloc[0, 2] if not pd.isna(usage_time.iloc[0, 2]) else datetime.now().strftime(
                        '%Y-%m-%d')

                    # Process app usage time data
                    if not usage_time.empty:
                        apps_df = usage_time.iloc[1:18, [0, 2, 3]].copy()
                        apps_df.columns = ['app_name', 'date', 'usage_time']
                        apps_df['file'] = file
                        apps_df['user_id'] = user_id
                        all_data.append(apps_df)

                    # Process unlock data if available
                    if not unlocks.empty:
                        unlock_count = unlocks.iloc[1, 2] if len(unlocks) > 1 and len(unlocks.columns) > 2 else 0
                        if user_id not in self.user_metrics:
                            self.user_metrics[user_id] = {}
                        self.user_metrics[user_id]['unlock_count'] = unlock_count

                except Exception as e:
                    print(f"Error processing {file}: {e}")

        # Combine all data
        if all_data:
            self.usage_data = pd.concat(all_data, ignore_index=True)
            print(f"Loaded data from {len(all_data)} files")
            return True
        return False

    def _extract_user_id(self, filename):
        """Extract user ID from filename"""
        # Attempt to extract user ID from filename
        # This is a simple implementation - adjust based on your actual filename format
        parts = filename.split('_')
        if len(parts) > 1:
            # Try to extract user_id from format like "user_123.xlsx"
            user_part = parts[1].split('.')[0]
            return user_part
        else:
            # If no clear pattern, use the filename without extension as user_id
            return os.path.splitext(filename)[0]

    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        if self.usage_data is None:
            print("No data loaded")
            return False

        # Convert usage time to seconds
        def parse_time(time_str):
            if pd.isna(time_str):
                return 0

            # Handle different formats (1m 11s, 58m 10s, 35s, etc.)
            total_seconds = 0
            if isinstance(time_str, str):
                parts = time_str.split()
                for part in parts:
                    if part.endswith('m'):
                        total_seconds += int(part[:-1]) * 60
                    elif part.endswith('s'):
                        total_seconds += int(part[:-1])
                    elif part.isdigit():
                        total_seconds += int(part)
            elif isinstance(time_str, (int, float)):
                total_seconds = time_str

            return total_seconds

        # Apply time parsing
        self.usage_data['usage_seconds'] = self.usage_data['usage_time'].apply(parse_time)

        # Categorize apps (simplified version)
        productivity_apps = ['docs.google.com', 'Drive', 'forms.gle', 'Canvas Student']
        social_apps = ['Instagram', 'WhatsApp Business']
        entertainment_apps = ['RYX Music', 'YouTube']

        def categorize_app(app_name):
            if app_name in productivity_apps:
                return 'productivity'
            elif app_name in social_apps:
                return 'social'
            elif app_name in entertainment_apps:
                return 'entertainment'
            else:
                return 'other'

        self.usage_data['category'] = self.usage_data['app_name'].apply(categorize_app)

        # Create features for ML
        # Group by file and calculate aggregate metrics
        file_metrics = self.usage_data.groupby(['file', 'user_id']).agg({
            'usage_seconds': ['sum', 'mean', 'max'],
            'app_name': 'count'
        })

        file_metrics.columns = ['total_usage', 'avg_usage', 'max_usage', 'app_count']
        file_metrics.reset_index(inplace=True)

        # Calculate category percentages
        category_usage = self.usage_data.groupby(['file', 'user_id', 'category'])['usage_seconds'].sum().unstack(
            fill_value=0)

        # If any categories are missing, add them with zeros
        for cat in ['productivity', 'social', 'entertainment', 'other']:
            if cat not in category_usage.columns:
                category_usage[cat] = 0

        # Calculate distraction ratio
        category_usage['distraction_ratio'] = (category_usage['social'] + category_usage['entertainment']) / \
                                              (category_usage['productivity'] + category_usage['social'] +
                                               category_usage['entertainment'] + category_usage['other'])

        # Replace NaN with 0
        category_usage.fillna(0, inplace=True)
        category_usage.reset_index(inplace=True)

        # Combine metrics
        self.features = pd.merge(file_metrics, category_usage, on=['file', 'user_id'])

        # Add distraction flag (usage > 3 hours or 10800 seconds)
        self.features['distraction_flag'] = (self.features['total_usage'] > 10800).astype(int)

        # Process per-user data
        self._process_user_data()

        print(f"Processed {len(self.features)} unique usage sessions for {len(self.user_metrics)} users")
        return True

    def _process_user_data(self):
        """Process data per user for individual analysis"""
        # Group by user_id
        user_grouped = self.usage_data.groupby('user_id')

        # For each user, calculate metrics
        for user_id, user_data in user_grouped:
            if user_id not in self.user_data:
                self.user_data[user_id] = {}

            self.user_data[user_id]['raw_data'] = user_data

            # Calculate key metrics for this user
            user_metrics = {
                'total_usage_seconds': user_data['usage_seconds'].sum(),
                'avg_session_seconds': user_data['usage_seconds'].mean(),
                'app_count': user_data['app_name'].nunique(),
                'top_apps': user_data.groupby('app_name')['usage_seconds'].sum().sort_values(ascending=False).head(
                    5).to_dict(),
                'category_usage': user_data.groupby('category')['usage_seconds'].sum().to_dict()
            }

            # Calculate distraction ratio
            category_totals = user_data.groupby('category')['usage_seconds'].sum()
            if 'productivity' not in category_totals:
                category_totals['productivity'] = 0
            if 'social' not in category_totals:
                category_totals['social'] = 0
            if 'entertainment' not in category_totals:
                category_totals['entertainment'] = 0
            if 'other' not in category_totals:
                category_totals['other'] = 0

            user_metrics['distraction_ratio'] = (category_totals.get('social', 0) + category_totals.get('entertainment',
                                                                                                        0)) / \
                                                (category_totals.sum() if category_totals.sum() > 0 else 1)

            # Store metrics
            self.user_metrics[user_id] = {**self.user_metrics.get(user_id, {}), **user_metrics}

    def run_clustering(self, n_clusters=3):
        """Perform K-means clustering on usage patterns"""
        if self.features is None or len(self.features) < n_clusters:
            print("Not enough data for clustering")
            return False

        # Select and scale features for clustering
        cluster_features = ['total_usage', 'avg_usage', 'app_count',
                            'productivity', 'social', 'entertainment', 'distraction_ratio']

        X = self.features[cluster_features].copy()

        # Handle missing values
        X.fillna(0, inplace=True)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        self.features['cluster'] = cluster_labels

        # Add cluster info to user metrics
        for i, row in self.features.iterrows():
            user_id = row['user_id']
            if user_id in self.user_metrics:
                if 'clusters' not in self.user_metrics[user_id]:
                    self.user_metrics[user_id]['clusters'] = []

                self.user_metrics[user_id]['clusters'].append(int(row['cluster']))

        # Store most common cluster for each user
        for user_id in self.user_metrics:
            if 'clusters' in self.user_metrics[user_id]:
                clusters = self.user_metrics[user_id]['clusters']
                self.user_metrics[user_id]['primary_cluster'] = max(set(clusters),
                                                                    key=clusters.count) if clusters else -1

        # Calculate silhouette score if enough samples
        if len(X_scaled) > n_clusters:
            silhouette = silhouette_score(X_scaled, self.features['cluster'])
            print(f"Clustering complete - Silhouette Score: {silhouette:.3f}")
        else:
            print("Clustering complete (insufficient samples for silhouette score)")

        self.models['kmeans'] = kmeans
        return True

    def train_predictive_models(self):
        """Train logistic regression, decision tree and random forest models"""
        if self.features is None or len(self.features) < 10:
            print("Not enough data for training models")
            return False

        # Prepare features and target
        model_features = ['total_usage', 'avg_usage', 'max_usage', 'app_count',
                          'productivity', 'social', 'entertainment', 'other', 'distraction_ratio']

        X = self.features[model_features].copy()
        y = self.features['distraction_flag']

        # Handle missing values
        X.fillna(0, inplace=True)

        # Split data if enough samples
        if len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Train logistic regression
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)
        self.models['logistic'] = log_reg

        # Train decision tree
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt.fit(X_train, y_train)
        self.models['decision_tree'] = dt

        # Train random forest
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        # Evaluate models
        if len(X_test) > 1:
            for name, model in self.models.items():
                if name not in ['kmeans']:
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    print(f"{name} - Accuracy: {acc:.3f}, F1 Score: {f1:.3f}")

        # Apply model predictions to user data
        for user_id in self.user_metrics:
            user_features = self.features[self.features['user_id'] == user_id]
            if not user_features.empty:
                user_X = user_features[model_features].copy()

                # Make predictions for this user if we have enough data
                if len(user_X) > 0:
                    self.user_metrics[user_id]['predictions'] = {
                        'logistic': int(self.models['logistic'].predict(user_X).mean() > 0.5),
                        'decision_tree': int(self.models['decision_tree'].predict(user_X).mean() > 0.5),
                        'random_forest': int(self.models['random_forest'].predict(user_X).mean() > 0.5)
                    }

                    # Calculate consensus prediction
                    predictions = list(self.user_metrics[user_id]['predictions'].values())
                    self.user_metrics[user_id]['distraction_prediction'] = int(
                        sum(predictions) / len(predictions) > 0.5)

        return True

    def generate_user_visualizations(self, user_id, output_dir):
        """Generate visualizations for a specific user"""
        if user_id not in self.user_data:
            return False

        user_output_dir = os.path.join(output_dir, f"user_{user_id}")
        os.makedirs(user_output_dir, exist_ok=True)

        user_data = self.user_data[user_id]['raw_data']

        # 1. App usage distribution
        plt.figure(figsize=(10, 6))
        app_usage = user_data.groupby('app_name')['usage_seconds'].sum().sort_values(ascending=False)
        sns.barplot(x=app_usage.values, y=app_usage.index)
        plt.title(f'User {user_id}: Total App Usage Time')
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(user_output_dir, 'app_usage_distribution.png'))
        plt.close()

        # 2. Category distribution
        plt.figure(figsize=(8, 8))
        category_usage = user_data.groupby('category')['usage_seconds'].sum()
        plt.pie(category_usage, labels=category_usage.index, autopct='%1.1f%%')
        plt.title(f'User {user_id}: Usage by Category')
        plt.savefig(os.path.join(user_output_dir, 'category_distribution.png'))
        plt.close()

        # 3. Daily usage pattern if date information is available
        if 'date' in user_data.columns and not user_data['date'].isna().all():
            try:
                daily_usage = user_data.groupby('date')['usage_seconds'].sum()
                plt.figure(figsize=(10, 6))
                daily_usage.plot(kind='bar')
                plt.title(f'User {user_id}: Daily Usage Pattern')
                plt.xlabel('Date')
                plt.ylabel('Usage (seconds)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(user_output_dir, 'daily_usage_pattern.png'))
                plt.close()
            except Exception as e:
                print(f"Could not create daily usage chart for user {user_id}: {e}")

        return True

    def generate_visualizations(self, output_dir):
        """Create and save visualizations"""
        if self.usage_data is None or self.features is None:
            print("No data for visualizations")
            return False

        os.makedirs(output_dir, exist_ok=True)

        # 1. App usage distribution
        plt.figure(figsize=(10, 6))
        app_usage = self.usage_data.groupby('app_name')['usage_seconds'].sum().sort_values(ascending=False)
        sns.barplot(x=app_usage.values, y=app_usage.index)
        plt.title('Total App Usage Time')
        plt.xlabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'app_usage_distribution.png'))
        plt.close()

        # 2. Category distribution
        plt.figure(figsize=(8, 8))
        category_usage = self.usage_data.groupby('category')['usage_seconds'].sum()
        plt.pie(category_usage, labels=category_usage.index, autopct='%1.1f%%')
        plt.title('Usage by Category')
        plt.savefig(os.path.join(output_dir, 'category_distribution.png'))
        plt.close()

        # 3. Cluster visualization (if clustering was performed)
        if 'cluster' in self.features.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                x='total_usage',
                y='distraction_ratio',
                hue='cluster',
                palette='viridis',
                data=self.features
            )
            plt.title('Usage Clusters')
            plt.xlabel('Total Usage Time (seconds)')
            plt.ylabel('Distraction Ratio')
            plt.savefig(os.path.join(output_dir, 'usage_clusters.png'))
            plt.close()

        # 4. Feature importance (if random forest model exists)
        if 'random_forest' in self.models:
            plt.figure(figsize=(10, 6))
            model_features = ['total_usage', 'avg_usage', 'max_usage', 'app_count',
                              'productivity', 'social', 'entertainment', 'other', 'distraction_ratio']
            importances = self.models['random_forest'].feature_importances_
            indices = np.argsort(importances)

            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [model_features[i] for i in indices])
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()

        # 5. User Comparison Charts
        self._generate_comparison_charts(output_dir)

        print(f"Saved visualizations to {output_dir}")
        return True

    def _generate_comparison_charts(self, output_dir):
        """Generate charts comparing different users"""
        if len(self.user_metrics) <= 1:
            print("Not enough users for comparison charts")
            return

        # Prepare data for comparison
        users = list(self.user_metrics.keys())
        total_usage = [self.user_metrics[user].get('total_usage_seconds', 0) / 3600 for user in
                       users]  # Convert to hours
        distraction_ratios = [self.user_metrics[user].get('distraction_ratio', 0) * 100 for user in
                              users]  # Convert to percentage

        # 1. Total Usage Comparison
        plt.figure(figsize=(10, 8))
        sns.barplot(x=users, y=total_usage)
        plt.title('Total Usage Comparison (Hours)')
        plt.xlabel('User ID')
        plt.ylabel('Total Hours')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_usage_comparison.png'))
        plt.close()

        # 2. Distraction Ratio Comparison
        plt.figure(figsize=(10, 8))
        sns.barplot(x=users, y=distraction_ratios)
        plt.title('Distraction Ratio Comparison (%)')
        plt.xlabel('User ID')
        plt.ylabel('Distraction Ratio (%)')
        plt.xticks(rotation=45)
        plt.axhline(y=30, color='r', linestyle='--', label='30% Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'user_distraction_comparison.png'))
        plt.close()

        # 3. Category Usage Comparison
        # Prepare data
        category_data = []
        for user in users:
            if 'category_usage' in self.user_metrics[user]:
                for category, seconds in self.user_metrics[user]['category_usage'].items():
                    category_data.append({
                        'user_id': user,
                        'category': category,
                        'minutes': seconds / 60  # Convert to minutes
                    })

        if category_data:
            category_df = pd.DataFrame(category_data)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='user_id', y='minutes', hue='category', data=category_df)
            plt.title('Category Usage by User (Minutes)')
            plt.xlabel('User ID')
            plt.ylabel('Minutes')
            plt.xticks(rotation=45)
            plt.legend(title='Category')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'category_comparison.png'))
            plt.close()

    def generate_user_report(self, user_id, output_dir):
        """Generate an HTML report for a specific user"""
        if user_id not in self.user_metrics:
            print(f"No data for user {user_id}")
            return False

        user_output_dir = os.path.join(output_dir, f"user_{user_id}")
        os.makedirs(user_output_dir, exist_ok=True)
        report_path = os.path.join(user_output_dir, 'usage_report.html')

        user_metrics = self.user_metrics[user_id]

        # Calculate percentile rankings compared to other users
        if len(self.user_metrics) > 1:
            all_usage = [self.user_metrics[u].get('total_usage_seconds', 0) for u in self.user_metrics]
            all_distraction = [self.user_metrics[u].get('distraction_ratio', 0) for u in self.user_metrics]

            usage_percentile = np.percentile(all_usage, np.searchsorted(np.sort(all_usage),
                                                                        user_metrics.get('total_usage_seconds',
                                                                                         0)) / len(all_usage) * 100)
            distraction_percentile = np.percentile(all_distraction, np.searchsorted(np.sort(all_distraction),
                                                                                    user_metrics.get(
                                                                                        'distraction_ratio', 0)) / len(
                all_distraction) * 100)
        else:
            usage_percentile = 50
            distraction_percentile = 50

        # Generate recommendations based on metrics
        recommendations = []

        if user_metrics.get('distraction_ratio', 0) > 0.3:
            recommendations.append(
                "Your distraction ratio is high. Consider setting time limits for social and entertainment apps.")

        if user_metrics.get('total_usage_seconds', 0) > 14400:  # 4 hours
            recommendations.append(
                "Your total screen time is above average. Try to take regular breaks and set daily usage limits.")

        if user_metrics.get('unlock_count', 0) > 50:
            recommendations.append(
                "You unlock your device frequently. Consider enabling 'Do Not Disturb' mode during focused work periods.")

        if len(recommendations) == 0:
            recommendations.append("Your usage patterns look healthy! Keep up the good work.")

        # Generate HTML
        html = f"""
        <html>
        <head>
            <title>App Usage Analysis Report - User {user_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .highlight {{ color: #d9534f; }}
                .good {{ color: #5cb85c; }}
                .warning {{ color: #f0ad4e; }}
                .danger {{ color: #d9534f; }}
                .percentile {{ font-size: 0.9em; color: #666; }}
                .recommendations {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .comparison {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>App Usage Analysis Report - User {user_id}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

            <h2>Key Metrics</h2>
            <div class="metric">Total Usage Time: <b>{user_metrics.get('total_usage_seconds', 0) / 3600:.2f} hours</b>
                <span class="percentile">(Percentile: {usage_percentile:.0f}%)</span>
            </div>
            <div class="metric">Average Session: <b>{user_metrics.get('avg_session_seconds', 0):.2f} seconds</b></div>
            <div class="metric">Unique Apps Used: <b>{user_metrics.get('app_count', 0)}</b></div>
            <div class="metric">Distraction Ratio: <b class="{
        'good' if user_metrics.get('distraction_ratio', 0) < 0.2 else
        'warning' if user_metrics.get('distraction_ratio', 0) < 0.4 else
        'danger'
        }">{user_metrics.get('distraction_ratio', 0) * 100:.1f}%</b>
                <span class="percentile">(Percentile: {distraction_percentile:.0f}%)</span>
            </div>
            <div class="metric">Device Unlocks: <b>{user_metrics.get('unlock_count', 'N/A')}</b></div>
        """

        # Add cluster and prediction information if available
        if 'primary_cluster' in user_metrics and user_metrics['primary_cluster'] >= 0:
            cluster_descriptions = {
                0: "Low usage / productivity focused",
                1: "Moderate usage / balanced",
                2: "High usage / distraction heavy"
            }

            cluster_id = user_metrics['primary_cluster']
            description = cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")

            html += f"""
            <div class="metric">Usage Pattern: <b>{description}</b></div>
            """

        if 'distraction_prediction' in user_metrics:
            prediction = "High risk of distraction" if user_metrics[
                'distraction_prediction'] else "Low risk of distraction"
            prediction_class = "danger" if user_metrics['distraction_prediction'] else "good"

            html += f"""
            <div class="metric">Distraction Prediction: <b class="{prediction_class}">{prediction}</b></div>
            """

        # Add top apps
        html += """
            <h2>Top Apps by Usage</h2>
            <ul>
        """

        if 'top_apps' in user_metrics:
            for app, usage in user_metrics['top_apps'].items():
                minutes = usage / 60
                html += f"<li><b>{app}</b>: {minutes:.1f} minutes</li>\n"

        html += """
            </ul>

            <h2>Visualizations</h2>
            <img src="app_usage_distribution.png" width="600">
            <img src="category_distribution.png" width="400">
        """

        # Add daily pattern if it exists
        if os.path.exists(os.path.join(user_output_dir, 'daily_usage_pattern.png')):
            html += """
            <img src="daily_usage_pattern.png" width="600">
            """

        # Add recommendations section
        html += """
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
        """

        for recommendation in recommendations:
            html += f"<li>{recommendation}</li>\n"

        html += """
                </ul>
            </div>
        """

        # Add comparison section if we have multiple users
        if len(self.user_metrics) > 1:
            # Calculate rankings
            all_users = list(self.user_metrics.keys())
            usage_ranking = sorted(all_users, key=lambda u: self.user_metrics[u].get('total_usage_seconds', 0))
            distraction_ranking = sorted(all_users, key=lambda u: self.user_metrics[u].get('distraction_ratio', 0))

            user_usage_rank = usage_ranking.index(user_id) + 1
            user_distraction_rank = distraction_ranking.index(user_id) + 1

            html += f"""
            <h2>How You Compare</h2>
            <div class="comparison">
                <p>Out of {len(all_users)} users:</p>
                <ul>
                    <li>You rank <b>#{user_usage_rank}</b> in total usage time</li>
                    <li>You rank <b>#{user_distraction_rank}</b> in distraction ratio</li>
                </ul>
                <p>View the overall comparison report for more details on how you compare to other users.</p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html)

        print(f"User report generated at {report_path}")
        return True

    def generate_comparison_report(self, output_dir):
        """Generate a comparison report for all users"""
        if len(self.user_metrics) <= 1:
            print("Not enough users for comparison report")
            return False

        report_path = os.path.join(output_dir, 'user_comparison_report.html')

        # Prepare comparison data
        users = list(self.user_metrics.keys())
        user_data = []

        for user_id in users:
            metrics = self.user_metrics[user_id]
            user_data.append({
                'user_id': user_id,
                'total_hours': metrics.get('total_usage_seconds', 0) / 3600,
                'distraction_ratio': metrics.get('distraction_ratio', 0) * 100,
                'app_count': metrics.get('app_count', 0),
                'avg_session': metrics.get('avg_session_seconds', 0),
                'unlocks': metrics.get('unlock_count', 'N/A'),
                'cluster': metrics.get('primary_cluster', -1),
                'distraction_prediction': metrics.get('distraction_prediction', None)
            })

        # Sort users by total usage
        user_data.sort(key=lambda x: x['total_hours'], reverse=True)

        # Generate HTML
        html = f"""
                <html>
                <head>
                    <title>User Comparison Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        tr:hover {{ background-color: #f5f5f5; }}
                        .good {{ color: #5cb85c; }}
                        .warning {{ color: #f0ad4e; }}
                        .danger {{ color: #d9534f; }}
                        .chart-container {{ margin: 30px 0; }}
                        .insights {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                        .links {{ margin-top: 20px; }}
                    </style>
                </head>
                <body>
                    <h1>User Comparison Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

                    <div class="insights">
                        <h2>Key Insights</h2>
                        <ul>
                            <li>Average total usage across all users: <b>{np.mean([u['total_hours'] for u in user_data]):.2f} hours</b></li>
                            <li>Average distraction ratio: <b>{np.mean([u['distraction_ratio'] for u in user_data]):.1f}%</b></li>
                            <li>User with highest usage: <b>{user_data[0]['user_id']}</b> with <b>{user_data[0]['total_hours']:.2f} hours</b></li>
                            <li>User with lowest distraction ratio: <b>{min(user_data, key=lambda x: x['distraction_ratio'])['user_id']}</b> with 
                                <b>{min(user_data, key=lambda x: x['distraction_ratio'])['distraction_ratio']:.1f}%</b></li>
                        </ul>
                    </div>

                    <h2>User Comparison Table</h2>
                    <table>
                        <tr>
                            <th>User ID</th>
                            <th>Total Usage (hours)</th>
                            <th>Distraction Ratio</th>
                            <th>App Count</th>
                            <th>Avg Session (sec)</th>
                            <th>Unlocks</th>
                            <th>Usage Pattern</th>
                            <th>Distraction Risk</th>
                        </tr>
                """

        # Define cluster descriptions
        cluster_descriptions = {
            0: "Low usage / productivity focused",
            1: "Moderate usage / balanced",
            2: "High usage / distraction heavy"
        }

        # Add rows for each user
        for user in user_data:
            # Determine CSS classes based on values
            distraction_class = 'good' if user['distraction_ratio'] < 20 else 'warning' if user[
                                                                                               'distraction_ratio'] < 40 else 'danger'

            # Get cluster description
            cluster_desc = cluster_descriptions.get(user['cluster'], "Unknown") if user['cluster'] >= 0 else "Unknown"

            # Format distraction prediction
            if user['distraction_prediction'] is not None:
                prediction = "High risk" if user['distraction_prediction'] else "Low risk"
                pred_class = "danger" if user['distraction_prediction'] else "good"
            else:
                prediction = "Unknown"
                pred_class = ""

            html += f"""
                        <tr>
                            <td><a href="user_{user['user_id']}/usage_report.html">{user['user_id']}</a></td>
                            <td>{user['total_hours']:.2f}</td>
                            <td class="{distraction_class}">{user['distraction_ratio']:.1f}%</td>
                            <td>{user['app_count']}</td>
                            <td>{user['avg_session']:.1f}</td>
                            <td>{user['unlocks']}</td>
                            <td>{cluster_desc}</td>
                            <td class="{pred_class}">{prediction}</td>
                        </tr>
                    """

        html += """
                    </table>

                    <h2>Comparison Charts</h2>
                    <div class="chart-container">
                        <h3>Total Usage Comparison</h3>
                        <img src="user_usage_comparison.png" width="800">
                    </div>

                    <div class="chart-container">
                        <h3>Distraction Ratio Comparison</h3>
                        <img src="user_distraction_comparison.png" width="800">
                    </div>

                    <div class="chart-container">
                        <h3>Category Usage Comparison</h3>
                        <img src="category_comparison.png" width="800">
                    </div>

                    <div class="links">
                        <h2>Individual User Reports</h2>
                        <ul>
                """

        # Add links to individual user reports
        for user_id in users:
            html += f"""
                        <li><a href="user_{user_id}/usage_report.html">User {user_id} Report</a></li>
                    """

        html += """
                        </ul>
                    </div>
                </body>
                </html>
                """

        with open(report_path, 'w') as f:
            f.write(html)

        print(f"Comparison report generated at {report_path}")
        return True

    def export_metrics(self, output_dir):
        """Export metrics as JSON for possible later use"""
        metrics_path = os.path.join(output_dir, 'user_metrics.json')

        # Prepare metrics for JSON serialization
        serializable_metrics = {}

        for user_id, metrics in self.user_metrics.items():
            serializable_metrics[user_id] = {}
            for key, value in metrics.items():
                # Convert numpy types and other non-serializable types
                if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    serializable_metrics[user_id][key] = float(value)
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like top_apps)
                    serializable_metrics[user_id][key] = {
                        k: float(v) if isinstance(v, (np.int64, np.int32, np.float64, np.float32)) else v
                        for k, v in value.items()}
                elif isinstance(value, list):
                    # Handle lists
                    serializable_metrics[user_id][key] = [
                        float(x) if isinstance(x, (np.int64, np.int32, np.float64, np.float32)) else x
                        for x in value]
                else:
                    serializable_metrics[user_id][key] = value

        # Save to JSON
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Metrics exported to {metrics_path}")
        return True

    def generate_report(self, output_dir):
        """Generate an overall HTML report with findings"""
        if self.usage_data is None or self.features is None:
            print("No data for report generation")
            return False

        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'overall_usage_report.html')

        # Calculate key metrics
        total_usage = self.features['total_usage'].sum()
        avg_session = self.features['avg_usage'].mean()
        distraction_ratio = self.features['distraction_ratio'].mean()

        # Top apps by usage
        top_apps = self.usage_data.groupby('app_name')['usage_seconds'].sum().sort_values(ascending=False).head(5)

        # Generate HTML
        html = f"""
        <html>
        <head>
            <title>Overall App Usage Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .highlight {{ color: #d9534f; }}
                .links {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Overall App Usage Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p>This report contains aggregate information across all {len(self.user_metrics)} users in the dataset.</p>

            <h2>Key Metrics</h2>
            <div class="metric">Total Usage Time: <b>{total_usage / 3600:.2f} hours</b></div>
            <div class="metric">Average Session: <b>{avg_session:.2f} seconds</b></div>
            <div class="metric">Average Distraction Ratio: <b>{distraction_ratio * 100:.1f}%</b></div>

            <h2>Top 5 Apps by Usage</h2>
            <ul>
        """

        for app, usage in top_apps.items():
            minutes = usage / 60
            html += f"<li><b>{app}</b>: {minutes:.1f} minutes</li>\n"

        html += """
            </ul>

            <h2>Visualizations</h2>
            <img src="app_usage_distribution.png" width="600">
            <img src="category_distribution.png" width="400">
            <img src="usage_clusters.png" width="600">
            <img src="feature_importance.png" width="600">

            <h2>Model Performance</h2>
            <p>The following models were trained to predict high distraction patterns:</p>
            <ul>
        """

        # Add model performance metrics if available
        if len(self.models) > 0:
            for name, model in self.models.items():
                if name == 'kmeans':
                    continue  # Skip clustering model
                html += f"<li><b>{name}</b></li>\n"
        else:
            html += "<li>No models trained (insufficient data)</li>\n"

        html += """
            </ul>

            <h2>Recommendations</h2>
            <ul>
                <li>Consider setting time limits for apps with high usage.</li>
                <li>Aim for a distraction ratio under 30% for better productivity.</li>
                <li>Try to limit device unlocks to less than 50 per day.</li>
            </ul>

            <div class="links">
                <h2>Additional Reports</h2>
                <ul>
                    <li><a href="user_comparison_report.html">User Comparison Report</a></li>
        """

        # Add links to individual user reports
        for user_id in self.user_metrics:
            html += f"""
                    <li><a href="user_{user_id}/usage_report.html">User {user_id} Report</a></li>
            """

        html += """
                </ul>
            </div>
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html)

        print(f"Overall report generated at {report_path}")
        return True

    def run_pipeline(self, output_dir):
        """Run the complete analysis pipeline"""
        print("Starting App Usage Analysis Pipeline")

        if not self.load_data():
            print("Failed to load data. Exiting pipeline.")
            return False

        if not self.preprocess_data():
            print("Failed to preprocess data. Exiting pipeline.")
            return False

        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run ML models and generate overall visualizations
        self.run_clustering()
        self.train_predictive_models()
        self.generate_visualizations(output_dir)

        # Generate individual user reports and visualizations
        for user_id in self.user_metrics:
            self.generate_user_visualizations(user_id, output_dir)
            self.generate_user_report(user_id, output_dir)

        # Generate comparison report if we have multiple users
        if len(self.user_metrics) > 1:
            self.generate_comparison_report(output_dir)

        # Generate overall report and export metrics
        self.generate_report(output_dir)
        self.export_metrics(output_dir)

        # Save model artifacts if needed
        self._save_models(output_dir)

        print("Pipeline completed successfully!")
        return True

    def _save_models(self, output_dir):
        """Save trained models for future use"""
        import pickle

        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Save each model
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        print(f"Models saved to {models_dir}")
        return True


class MLOpsManager:
    """Manages the offline ML pipeline for app usage analysis"""

    def __init__(self, data_dir, output_base_dir):
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.analyzer = None
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_output_dir = None

    def setup_dirs(self):
        """Set up the directories for this run"""
        # Create timestamped output directory
        self.current_output_dir = os.path.join(self.output_base_dir, f"run_{self.run_timestamp}")
        os.makedirs(self.current_output_dir, exist_ok=True)

        # Create logs directory
        logs_dir = os.path.join(self.current_output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        return True

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print(f"Starting ML pipeline run at {self.run_timestamp}")

        # Setup directories
        self.setup_dirs()

        # Configure logging
        log_file = os.path.join(self.current_output_dir, 'logs', 'pipeline.log')
        self._setup_logging(log_file)

        try:
            # Create and run analyzer
            self.analyzer = AppUsageAnalyzer(self.data_dir)
            success = self.analyzer.run_pipeline(self.current_output_dir)

            if success:
                print(f"Pipeline run completed successfully. Output saved to {self.current_output_dir}")
                # Create a success flag file
                with open(os.path.join(self.current_output_dir, 'success.flag'), 'w') as f:
                    f.write(f"Pipeline completed successfully at {datetime.now()}")
            else:
                print("Pipeline run failed")
                # Create a failure flag file
                with open(os.path.join(self.current_output_dir, 'failure.flag'), 'w') as f:
                    f.write(f"Pipeline failed at {datetime.now()}")

            return success

        except Exception as e:
            print(f"Error in pipeline run: {e}")
            import traceback
            with open(os.path.join(self.current_output_dir, 'logs', 'error.log'), 'w') as f:
                f.write(f"Error at {datetime.now()}:\n")
                f.write(traceback.format_exc())
            return False

    def _setup_logging(self, log_file):
        """Set up logging to file and console"""
        import logging

        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.info(f"Logging initialized for pipeline run {self.run_timestamp}")
        return True


if __name__ == "__main__":
    # Set the directory containing Excel files and output directory
    data_dir = "./data"
    output_base_dir = "./output"

    # Create and run the ML pipeline
    mlops = MLOpsManager(data_dir, output_base_dir)
    mlops.run_analysis()


