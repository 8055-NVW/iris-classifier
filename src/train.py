#Imports
import argparse
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main(test_size: float = 0.20, random_state: int = 42, max_depth: int | None = 3):
    #Load and Inspect Dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species_id"] = y
    df["species"] = pd.Categorical.from_codes(y, iris.target_names)

    print("shape:", df.shape)
    print(df.head().to_string(index=False))
    print("species counts:\n", df["species"].value_counts().to_string())

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,     
        random_state=random_state,
        stratify=y
    )

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)
    print("train class counts:", np.bincount(y_train))
    print("test  class counts:", np.bincount(y_test))

    #Check for most optimum depth
    print("\nResult for different tree depths:")
    depths = [2, 3, 4, 5]
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=random_state)
        m.fit(X_train, y_train)
        print(f"max_depth={d:<5}  test_acc={m.score(X_test, y_test):.3f}")

    #Train Model
    model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)

    #Comparing the accuracy of predictions(the models results) to true labels
    y_pred = model.predict(X_test)
    print("\nPredictions:", y_pred[:15])
    print("True Labels:", y_test[:15])

    #Check for accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy, 3))

    #Export trained model
    joblib.dump(model, "outputs/model.joblib")
    print("Saved trained model to outputs/model.joblib")

    #Confusion matrix
    Path("outputs").mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Decision Tree — Confusion Matrix")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=150)
    print("Saved: outputs/confusion_matrix.png")
    plt.show()

    #Viewing the most important features used in the decision tree
    imp = pd.Series(model.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
    print("\nFeature importances:\n", imp.to_string())

    #Creating a tree structure to visualise the decisions
    plt.figure(figsize=(6, 6))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.tight_layout()
    plt.show()

    #Tested model with k-Nearest Neighbors (k-NN) algorithm
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    print("\nk-NN accuracy:", round(accuracy_score(y_test, y_pred_knn), 3))

    #Tested model with Support Vector Machine (SVM) algorithm
    model_svc = SVC(C=2, kernel="rbf", gamma="scale", random_state=random_state)
    model_svc.fit(X_train, y_train)
    y_pred_svc = model_svc.predict(X_test)
    print("SVM accuracy:", round(accuracy_score(y_test, y_pred_svc), 3))


if __name__ == "__main__":
    # simple CLI to match your course requirement
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split (e.g., 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-depth", type=int, default=3, help="Decision tree max depth (e.g., 3)")
    args = parser.parse_args()

    main(test_size=args.test_size, random_state=args.random_state, max_depth=args.max_depth)
