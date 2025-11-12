#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#Load and Inspect Dataset
iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df["species_id"] = y
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

print("shape:", df.shape)
display(df.head())
print("species counts:\n", df["species"].value_counts())

print("Result for most optimum depth")
depths = [2, 3, 4, 5]
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    print(f"max_depth={d:<5}  test_acc={m.score(X_test, y_test):.3f}")

#Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,     
    random_state=42,      
    stratify=y 
)

model = DecisionTreeClassifier(random_state = 42, max_depth=3)
model.fit(X_train, y_train)

#Prediction Check
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:15])
print("True Labels:", y_test[:15])

#Accuracy check
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(4,4))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Decision Tree — Confusion Matrix")
plt.tight_layout()
plt.show()

#View most important features usiend in the decision tree
imp = pd.Series(model.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
print(imp)

#Create tree structure to visualise the decisions
plt.figure(figsize=(6,6))
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