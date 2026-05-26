# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Page configuration
st.set_page_config(
    page_title="SVM Classification",
    layout="wide"
)

# Custom styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    h1 {
        color: white;
        text-align: center;
        font-size: 45px;
    }

    h3 {
        color: #4CAF50;
    }

    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .css-1d391kg {
        background-color: #161B22;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Student Performance Prediction Using SVM")

# Subtitle
st.markdown(
    "<center>Support Vector Machine Classification Model</center>",
    unsafe_allow_html=True
)

# Read dataset
df = pd.read_csv("student-por.csv")

# Remove duplicates
df = df.drop_duplicates()

# Create pass/fail column
df["result"] = np.where(df["G3"] >= 10, 1, 0)

# Store encoders
encoders = {}

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

# Categorical columns
cat_cols = df.select_dtypes(include="object").columns

# Fill missing numerical values
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill missing categorical values
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Encode categorical columns
for col in cat_cols:

    # Create encoder
    le = LabelEncoder()

    # Encode values
    df[col] = le.fit_transform(df[col])

    # Store encoder
    encoders[col] = le

# Create features and target
X = df.drop("result", axis=1)
y = df["result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale dataset
scaler = StandardScaler()

# Fit train data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# Create SVM model
model = SVC(
    kernel="rbf",
    probability=True
)

# Train model
model.fit(X_train_scaled, y_train)

# Predict output
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Metrics section
st.subheader("Model Performance")

# Create columns
c1, c2, c3 = st.columns(3)

# Show metrics
c1.metric("Accuracy", f"{round(accuracy * 100, 2)} %")
c2.metric("Training Rows", X_train.shape[0])
c3.metric("Testing Rows", X_test.shape[0])

# Confusion matrix section
st.subheader("Confusion Matrix")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create figure
fig, ax = plt.subplots(figsize=(5, 5))

# Plot matrix
ax.imshow(cm)

# Axis labels
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Add values inside matrix
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )

# Show graph
st.pyplot(fig)

# Classification report
st.subheader("Classification Report")

# Generate report
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

# Convert report into dataframe
report_df = pd.DataFrame(report).transpose()

# Show report
st.dataframe(report_df)

# Prediction section
st.subheader("Predict Student Result")

# Store user inputs
user_input = {}

# Create two columns
col1, col2 = st.columns(2)

# Numerical features
with col1:

    # Loop numerical columns
    for col in num_cols:

        # Skip target columns
        if col not in ["result"]:

            # Create slider
            user_input[col] = st.slider(
                f"{col}",
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

# Categorical features
with col2:

    # Loop categorical columns
    for col in cat_cols:

        # Get original labels
        options = list(encoders[col].classes_)

        # Create selectbox
        selected = st.selectbox(
            f"{col}",
            options
        )

        # Encode selected value
        user_input[col] = encoders[col].transform(
            [selected]
        )[0]

# Predict button
if st.button("Predict Student Result"):

    # Convert into dataframe
    input_df = pd.DataFrame([user_input])

    # Arrange columns properly
    input_df = input_df[X.columns]

    # Scale values
    input_scaled = scaler.transform(input_df)

    # Predict class
    prediction = model.predict(input_scaled)

    # Predict probability
    probability = model.predict_proba(input_scaled)

    # Result box
    st.markdown("---")

    # Pass result
    if prediction[0] == 1:

        # Success result
        st.success(
            f"Prediction Result : PASS"
        )

    # Fail result
    else:

        # Error result
        st.error(
            f"Prediction Result : FAIL"
        )

    # Show probability
    st.info(
        f"Pass Probability : {round(probability[0][1] * 100, 2)} %"
    )