"""
Project: Predictive Modeling of Heart Disease

This version prints all outputs, including tables, statistics, and diagrams, to the console and displays plots, instead of generating an HTML report.

This script builds predictive models to determine whether a patient has heart disease or not, and provides a detailed report for each patient in the test set, including the model's prediction and probability.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

DEFAULT_FIGSIZE = (8, 5)
csv_file = './heart_disease.csv'

# Try to automatically detect the delimiter and handle bad lines
try:
    df = pd.read_csv(csv_file, comment='#', engine='python', on_bad_lines='skip')
except pd.errors.ParserError:
    try:
        df = pd.read_csv(csv_file, delimiter=';', comment='#', engine='python', on_bad_lines='skip')
    except pd.errors.ParserError:
        df = pd.read_csv(csv_file, delim_whitespace=True, comment='#', engine='python', on_bad_lines='skip')

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
buf = StringIO()
df.info(buf=buf)
print(buf.getvalue())

print("\nMissing Values per Column:")
print(df.isnull().sum())

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Identify the target column
target_col = None
for col in ['target', 'num']:
    if col in df.columns:
        target_col = col
        if col in numerical_cols:
            numerical_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
        break
if target_col is None:
    for col in df.columns:
        if df[col].nunique() == 2:
            target_col = col
            if col in numerical_cols:
                numerical_cols.remove(col)
            if col in categorical_cols:
                categorical_cols.remove(col)
            break
if target_col is None:
    raise ValueError("Target column not found.")

# Descriptive statistics
print("\nDescriptive Statistics (Numerical):")
desc_num = df[numerical_cols].describe()
print(desc_num)

if categorical_cols:
    print("\nDescriptive Statistics (Categorical):")
    desc_cat = df[categorical_cols].describe()
    print(desc_cat)
else:
    desc_cat = None

# Overlap count
overlap_count = 0
overlap_details = []
for col in numerical_cols:
    overlap_count += 2
    overlap_details.append(f"{col}: mean/std/min/max in both describe() and histogram")
    overlap_count += 4
    overlap_details.append(f"{col}: median/Q1/Q3/min/max in both describe() and boxplot")
    overlap_count += 2
    overlap_details.append(f"{col}: distribution/median in both describe() and violinplot")
if categorical_cols:
    for col in categorical_cols:
        overlap_count += 1
        overlap_details.append(f"{col}: frequency in both describe() and countplot")

print(f"\nOverlap of Information Between Description and Charts")
print(f"Overlap count: {overlap_count}")
for detail in overlap_details:
    print(f"- {detail}")

# Visualizations: Distribution of features
for col in numerical_cols:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

for col in categorical_cols:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    sns.countplot(x=col, data=df)
    plt.title(f'Countplot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=DEFAULT_FIGSIZE)
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numerical Features)')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Scatter plots for first three numerical features, colored by target
for i in range(min(3, len(numerical_cols))):
    for j in range(i+1, min(3, len(numerical_cols))):
        plt.figure(figsize=DEFAULT_FIGSIZE)
        sns.scatterplot(x=df[numerical_cols[i]], y=df[numerical_cols[j]], hue=df[target_col])
        plt.title(f'{numerical_cols[i]} vs {numerical_cols[j]}')
        plt.xlabel(numerical_cols[i])
        plt.ylabel(numerical_cols[j])
        plt.tight_layout()
        plt.show()

# Target variable distribution
print("\nTarget Variable Distribution:")
print(df[target_col].value_counts())
plt.figure(figsize=DEFAULT_FIGSIZE)
sns.countplot(x=target_col, data=df)
plt.title('Target Variable Distribution')
plt.xlabel(target_col)
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Boxplots and violin plots for each numerical feature against the target
for col in numerical_cols:
    plt.figure(figsize=DEFAULT_FIGSIZE)
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f'{col} by {target_col}')
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=DEFAULT_FIGSIZE)
    sns.violinplot(x=target_col, y=col, data=df)
    plt.title(f'{col} by {target_col} (Violin)')
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# ==========================================================
# Task 2: Data Preprocessing and Feature Engineering
# ==========================================================

imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')
df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
if categorical_cols:
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    feature_cols = [col for col in df.columns if col != target_col]
else:
    feature_cols = numerical_cols

if df[target_col].dtype == 'object' or str(df[target_col].dtype).startswith('category'):
    print(f"\nEncoding target column '{target_col}' with LabelEncoder")
    target_le = LabelEncoder()
    df[target_col] = target_le.fit_transform(df[target_col])
else:
    target_le = None

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# Task 3: Model Training and Development
# ==========================================================

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nModel Training")
print("Training Logistic Regression (GridSearch)...")
lr_params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
lr = GridSearchCV(LogisticRegression(max_iter=1000, solver='lbfgs'), lr_params, cv=cv, scoring='roc_auc', n_jobs=-1)
lr.fit(X_train, y_train)
results['Logistic Regression'] = lr.best_estimator_

print("Training KNN (GridSearch)...")
knn_params = {'n_neighbors': list(range(3, 11)), 'weights': ['uniform', 'distance']}
knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=cv, scoring='roc_auc', n_jobs=-1)
knn.fit(X_train, y_train)
results['KNN'] = knn.best_estimator_

print("Training SVM (GridSearch)...")
svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
svm = GridSearchCV(SVC(probability=True), svm_params, cv=cv, scoring='roc_auc', n_jobs=-1)
svm.fit(X_train, y_train)
results['SVM'] = svm.best_estimator_

print("Training Gradient Boosting (GridSearch)...")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 8]
}
gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=cv, scoring='roc_auc', n_jobs=-1)
gb.fit(X_train, y_train)
results['Gradient Boosting'] = gb.best_estimator_

print("Training Simple Feedforward Neural Network (PyTorch)...")
input_dim = X_train.shape[1]
hidden_dims = [64, 32]
output_dim = 1
dropout_rate = 0.2

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

mlp = SimpleMLP(input_dim, hidden_dims, output_dim, dropout_rate)
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

best_loss = np.inf
patience = 10
counter = 0
epochs = 100
losses = []
for epoch in range(epochs):
    mlp.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = mlp(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_state = mlp.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
mlp.load_state_dict(best_model_state)
results['Simple MLP'] = mlp

# Plot MLP loss curve
plt.figure(figsize=(8,4))
plt.plot(losses)
plt.title("MLP Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# ==========================================================
# Task 4: Model Evaluation and Detailed Patient Report
# ==========================================================

def evaluate_model(name, model, X_test, y_test):
    y_test_numeric = np.array(y_test).astype(np.float32)
    if name == 'Simple MLP':
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(torch.tensor(X_test.values, dtype=torch.float32)).numpy().flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_numeric, y_pred)
    prec = precision_score(y_test_numeric, y_pred)
    rec = recall_score(y_test_numeric, y_pred)
    f1 = f1_score(y_test_numeric, y_pred)
    auc = roc_auc_score(y_test_numeric, y_pred_prob)
    cm = confusion_matrix(y_test_numeric, y_pred)
    # Classification report as DataFrame
    cr = classification_report(y_test_numeric, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).T

    # Explicitly check and report all test cases: positive and negative
    positives = (y_test_numeric == 1)
    negatives = (y_test_numeric == 0)
    total_positives = np.sum(positives)
    total_negatives = np.sum(negatives)
    correct_positives = np.sum((y_pred == 1) & positives)
    correct_negatives = np.sum((y_pred == 0) & negatives)
    incorrect_positives = np.sum((y_pred == 0) & positives)
    incorrect_negatives = np.sum((y_pred == 1) & negatives)

    print(f"\n{name} Classification Report")
    print(cr_df)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]))
    print(f"Test Case Check for {name}:")
    print(f"  Total Positive Cases (Actual 1): {total_positives}")
    print(f"    Correctly Predicted Positive: {correct_positives}")
    print(f"    Incorrectly Predicted Negative (False Negatives): {incorrect_positives}")
    print(f"  Total Negative Cases (Actual 0): {total_negatives}")
    print(f"    Correctly Predicted Negative: {correct_negatives}")
    print(f"    Incorrectly Predicted Positive (False Positives): {incorrect_negatives}")

    # Detailed patient-level report
    print(f"\nDetailed Patient Prediction Report for {name}:")
    patient_report = pd.DataFrame(X_test, columns=X_test.columns)
    patient_report['Actual'] = y_test_numeric.astype(int)
    patient_report['Predicted'] = y_pred.astype(int)
    patient_report['Probability_Heart_Disease'] = y_pred_prob
    patient_report['Prediction_Label'] = np.where(patient_report['Predicted'] == 1, 'Heart Disease', 'No Heart Disease')
    patient_report['Actual_Label'] = np.where(patient_report['Actual'] == 1, 'Heart Disease', 'No Heart Disease')
    # Show first 10 patients as sample
    print(patient_report[['Actual_Label', 'Prediction_Label', 'Probability_Heart_Disease']].head(10))
    print(f"... ({len(patient_report)} patients in test set)")

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Confusion Matrix': cm,
        'Total Positives': total_positives,
        'Correct Positives': correct_positives,
        'Incorrect Positives (FN)': incorrect_positives,
        'Total Negatives': total_negatives,
        'Correct Negatives': correct_negatives,
        'Incorrect Negatives (FP)': incorrect_negatives,
        'Patient Report': patient_report
    }

print("\nModel Evaluation and Detailed Patient Reports")
summary = {}
for name, model in results.items():
    summary[name] = evaluate_model(name, model, X_test, y_test)

print("\nModel Performance Summary")
summary_df = pd.DataFrame({k: {metric: v[metric] for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']} for k, v in summary.items()}).T
print(summary_df)

# Confusion matrices
for name, metrics in summary.items():
    cm = metrics['Confusion Matrix']
    plt.figure(figsize=DEFAULT_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Identify best model by AUC-ROC
best_model_name = summary_df['AUC-ROC'].idxmax()
print(f"\nBest Performing Model: {best_model_name}")

# Show detailed report for best model (first 20 patients)
print(f"\nDetailed Patient Report for Best Model ({best_model_name}):")
best_patient_report = summary[best_model_name]['Patient Report']
print(best_patient_report[['Actual_Label', 'Prediction_Label', 'Probability_Heart_Disease']].head(20))
print(f"... ({len(best_patient_report)} patients in test set)")

# Feature importance for best traditional model
if best_model_name == 'Gradient Boosting':
    model = results[best_model_name]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        print(f"\nFeature Importances for {best_model_name}")
        print(feat_imp)
        plt.figure(figsize=DEFAULT_FIGSIZE)
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title(f'Feature Importances ({best_model_name})')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

elif best_model_name in ['Logistic Regression', 'SVM']:
    model = results[best_model_name]
    if hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        print(f"\nFeature Importances for {best_model_name}")
        print(feat_imp)
        plt.figure(figsize=DEFAULT_FIGSIZE)
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title(f'Feature Importances ({best_model_name})')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

# Permutation feature importance for Simple MLP
def permutation_importance(model, X, y, metric=accuracy_score, n_repeats=5):
    y_numeric = np.array(y).astype(np.float32)
    baseline = metric(y_numeric, (model(torch.tensor(X.values, dtype=torch.float32)).detach().numpy().flatten() > 0.5).astype(int))
    importances = []
    for col in X.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])
            score = metric(y_numeric, (model(torch.tensor(X_permuted.values, dtype=torch.float32)).detach().numpy().flatten() > 0.5).astype(int))
            scores.append(baseline - score)
        importances.append(np.mean(scores))
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\nPermutation Feature Importance for Simple MLP")
mlp_feat_imp = permutation_importance(results['Simple MLP'], X_test, y_test)
print(mlp_feat_imp)
plt.figure(figsize=DEFAULT_FIGSIZE)
sns.barplot(x=mlp_feat_imp.values, y=mlp_feat_imp.index)
plt.title('Feature Importances (Simple MLP)')
plt.xlabel('Importance (Decrease in Accuracy)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
