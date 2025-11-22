# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys

# Edit/add your Windows CSV path(s) here if the CSV is not in this project folder.
POSSIBLE_PATHS = [
    os.path.join(os.getcwd(), "HeartDiseaseTrain-Test.csv"),
    r"C:\Users\lavan\OneDrive\Desktop\Final\HeartDiseaseTrain-Test (4).csv",
    "/mnt/data/HeartDiseaseTrain-Test.csv",
    "/mnt/data/HeartDiseaseTrain-Test (11).csv",
]

def find_csv():
    for p in POSSIBLE_PATHS:
        if p and os.path.exists(p):
            return p
    print("CSV not found. Checked paths:")
    for p in POSSIBLE_PATHS:
        print("  -", p)
    print("\nFiles in current working directory:", os.getcwd())
    for f in os.listdir(os.getcwd()):
        print("   ", f)
    raise FileNotFoundError("Place HeartDiseaseTrain-Test.csv in this folder or modify POSSIBLE_PATHS in train_model.py")

def find_target_column(df):
    candidates = ['target','TARGET','heart_disease','HeartDisease','has_disease','cardio','cardio_event','HeartDiseaseYN','output']
    for c in candidates:
        if c in df.columns:
            return c
    last = df.columns[-1]
    if set(df[last].dropna().unique()).issubset({0,1,'0','1',True,False}):
        return last
    raise ValueError(f"Couldn't find a plausible target column. Columns: {list(df.columns)}")

def main():
    csv_path = find_csv()
    print("Using CSV:", csv_path)
    df = pd.read_csv(csv_path)
    print("CSV shape:", df.shape)

    target_col = find_target_column(df)
    print("Detected target column:", target_col)

    # Use numeric features only (safer automatic approach)
    X = df.select_dtypes(include=[np.number]).copy()
    if target_col in X.columns:
        y = X.pop(target_col)
    else:
        y = df[target_col]

    if len(set(y)) < 2:
        print("Target column has <2 classes. Aborting.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    print("Training on features:", list(X.columns))
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("Holdout accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    artifact = {'pipeline': pipeline, 'features': list(X.columns)}
    joblib.dump(artifact, "model.pkl")
    print("Saved model.pkl in", os.getcwd())

if __name__ == "__main__":
    main()
