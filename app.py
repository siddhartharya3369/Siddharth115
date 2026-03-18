
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Universal Bank - Personal Loan Prediction Dashboard")

df = pd.read_csv("dataset.csv")

st.subheader("Dataset Overview")
st.write(df.head())

# Features & target
X = df.drop(columns=["Personal Loan","ID","ZIPCode"])
y = df["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

plt.figure()

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append([name, acc, prec, rec, f1])

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=name)

st.subheader("Model Performance")
results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
st.dataframe(results_df)

plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
st.pyplot(plt)

# Confusion matrix
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
st.subheader("Confusion Matrix (Random Forest)")
st.pyplot(fig)

# Upload prediction
st.subheader("Upload Test Data")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    test = pd.read_csv(uploaded_file)
    preds = model.predict(test.drop(columns=["ID","ZIPCode"]))
    test["Predicted Loan"] = preds
    st.write(test.head())
    csv = test.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv")
