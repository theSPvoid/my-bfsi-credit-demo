import pandas as pd

# 1. Load the new file with alternative data
df = pd.read_csv("train_alt.csv")

# 2. Drop 'Loan_ID' if it exists (not needed for prediction)
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# 3. Convert Loan_Status from Y/N to 1/0
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# 4. Fill numeric missing values
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# 5. Our new alt data columns might have no missing (since we generated them),
#    but let's fill any potential missing values with mean to be safe
for col in ['Utility_Payment_Score','Mobile_Transactions','Social_Media_Score']:
    if col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

# 6. Fill categorical columns
for cat_col in ['Gender','Married','Dependents','Self_Employed','Property_Area','Education']:
    if cat_col in df.columns:
        df[cat_col].fillna(df[cat_col].mode()[0], inplace=True)

# 7. Convert '3+' in Dependents to numeric 3
if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# 8. Separate features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# 9. Create dummy variables for categorical fields
X = pd.get_dummies(X, drop_first=True)

# 10. Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Save to CSV
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data prep done. Created X_train.csv, X_test.csv, y_train.csv, y_test.csv.")