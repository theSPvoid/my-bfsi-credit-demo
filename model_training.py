import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load train/test data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Flatten y if needed
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# 2. Train Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
log_pred = logreg.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
print("Logistic Regression Accuracy:", log_acc)
print("Classification Report:\n", classification_report(y_test, log_pred))

# 3. Train Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
dt_pred = dtree.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_acc)
print("Classification Report:\n", classification_report(y_test, dt_pred))

# 4. Save models
joblib.dump(logreg, "logreg_model.pkl")
joblib.dump(dtree, "dtree_model.pkl")

# 5. Save feature names for dynamic explanation
feature_names = list(X_train.columns)
joblib.dump(feature_names, "feature_names.pkl")

print("Models trained & saved (logreg_model.pkl, dtree_model.pkl). Feature names saved in feature_names.pkl.")