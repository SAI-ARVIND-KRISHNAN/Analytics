from config import DATA_PATH
from data_loader import load_data
from eda import run_eda
from clustering import run_clustering
from encoding import encode_labels
from model_utils import train_and_save_models
from sklearn.model_selection import train_test_split

def main():
    df = load_data(DATA_PATH)
    run_eda(df)
    df = run_clustering(df)
    df, le = encode_labels(df)

    X = df[['total_time']]
    y = df['category_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_and_save_models(X_train, X_test, y_train, y_test, le)

if __name__ == "__main__":
    main()
