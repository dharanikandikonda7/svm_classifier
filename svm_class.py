# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Set page config
st.set_page_config(page_title="SVM Classification", layout="wide")

# Title
st.title(" Student Pass/Fail Prediction Using SVM")

# Read dataset
df = pd.read_csv("student-por.csv")

# Remove duplicates
df = df.drop_duplicates()

# Create target column
df["result"] = np.where(df["G3"] >= 10, 1, 0)

# Separate columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

# Fill missing numerical values
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill missing categorical values
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Create features and target
X = df.drop("result", axis=1)
y = df["result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model
model = SVC(kernel="linear")

# Train model
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Show accuracy
st.subheader("Model Accuracy")
st.success(f"Accuracy : {round(accuracy * 100, 2)} %")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots()

# Show matrix
ax.imshow(cm)

# Add labels
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Add values
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

# Show plot
st.pyplot(fig)