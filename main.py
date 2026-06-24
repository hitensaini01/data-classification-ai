import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_classification_pipeline():
    print("Initializing Data Classification Project...")
    
    # 1. Load dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    # 2. Split data
    X = df.drop(columns=['species'])
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scikit-Learn Pipeline
    model = KNeighborsClassifier(n_neighbors=5) # Instantiate
    model.fit(X_train, y_train)                  # Fit
    predictions = model.predict(X_test)          # Predict

    # 4. Evaluation
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n🎉 Model Accuracy: {accuracy * 100:.1f}%\n")
    print("--- Detailed Performance Breakdown ---")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

if __name__ == "__main__":
    run_classification_pipeline()
