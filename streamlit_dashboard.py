import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the cleaned data
data_path = 'cleaned_ecommerce_data.csv'
df = pd.read_csv(data_path)

# Title and description
st.title("E-commerce Data Dashboard")
st.markdown("""
This dashboard provides insights into e-commerce customer data, including purchase trends, product categories, payment methods, and churn analysis.
""")

# Sidebar filters
st.sidebar.header("Filters")
selected_category = st.sidebar.selectbox("Select Product Category", options=["All"] + df['product_category'].unique().tolist())
selected_payment_method = st.sidebar.selectbox("Select Payment Method", options=["All"] + df['payment_method'].unique().tolist())

# Apply filters
if selected_category != "All":
    df = df[df['product_category'] == selected_category]
if selected_payment_method != "All":
    df = df[df['payment_method'] == selected_payment_method]

# Purchases over time
st.subheader("Number of Purchases Over Time")
df['purchase_date_only'] = pd.to_datetime(df['purchase_date']).dt.date
daily_sales = df.groupby('purchase_date_only').size()

fig, ax = plt.subplots(figsize=(12, 5))
daily_sales.plot(ax=ax)
ax.set_title("Number of Purchases Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Purchases")
ax.grid(True)
st.pyplot(fig)
st.markdown("This graph shows the trend of purchases over time, helping to identify peak shopping periods.")

# Top product categories
st.subheader("Top Product Categories")
fig, ax = plt.subplots(figsize=(10, 5))
df['product_category'].value_counts().head(2).plot(kind='bar', ax=ax)
ax.set_title("Top 2 Product Categories")
ax.set_xlabel("Category")
ax.set_ylabel("Purchase Count")
st.pyplot(fig)
st.markdown("This bar chart highlights the most popular product categories based on purchase count.")

# Payment methods
st.subheader("Payment Method Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
df['payment_method'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
ax.set_title("Payment Method Distribution")
ax.set_ylabel("")
st.pyplot(fig)
st.markdown("This pie chart illustrates the distribution of payment methods used by customers.")

# Churn distribution
st.subheader("Churn Distribution")
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x='Churn', data=df, ax=ax)
ax.set_title("Churn Distribution")
ax.set_xlabel("Churn (0 = Active, 1 = Churned)")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)
st.markdown("This chart shows the distribution of churned versus active customers.")

# Returns distribution
st.subheader("Distribution of Returns")
df['returns_filled'] = df['Returns'].fillna(0)
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df['returns_filled'], bins=30, kde=True, color='blue', ax=ax)
ax.set_title("Distribution of Returns")
ax.set_xlabel("Returns")
ax.set_ylabel("Frequency")
st.pyplot(fig)
st.markdown("This histogram shows the frequency of returns across all transactions.")

# Returns by churn status
st.subheader("Returns by Churn Status")
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(x='Churn', y='returns_filled', data=df, ax=ax)
ax.set_title("Returns by Churn Status")
ax.set_xlabel("Churn")
ax.set_ylabel("Returns")
ax.set_ylim(0, 1)
st.pyplot(fig)
st.markdown("This boxplot compares the distribution of returns between churned and active customers.")

# SMOTE and Random Forest Analysis
st.sidebar.header("SMOTE and Random Forest Filters")
apply_smote = st.sidebar.checkbox("Apply SMOTE", value=False)

# Feature engineering
label_encoder = LabelEncoder()
df['Gender_encoded'] = label_encoder.fit_transform(df['Gender'].fillna('Unknown'))
df['payment_encoded'] = label_encoder.fit_transform(df['payment_method'].fillna('Unknown'))
df['category_encoded'] = label_encoder.fit_transform(df['product_category'].fillna('Unknown'))
df['has_returned'] = df['Returns'].fillna(0).apply(lambda x: 1 if x > 0 else 0)

# Select features and target
features = ['Gender_encoded', 'payment_encoded', 'category_encoded', 'has_returned']
target = 'Churn'
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if apply_smote:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, normalize='true')
ax.set_title("Confusion Matrix")
st.pyplot(fig)
st.markdown("This confusion matrix shows the performance of the Random Forest model.")

# ROC Curve
st.subheader("ROC Curve")
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax)
ax.set_title("ROC Curve")
st.pyplot(fig)
st.markdown("The ROC curve illustrates the trade-off between sensitivity and specificity for the Random Forest model.")

# Precision-Recall Curve
st.subheader("Precision-Recall Curve")
fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_estimator(rf_model, X_test, y_test, ax=ax)
ax.set_title("Precision-Recall Curve")
st.pyplot(fig)
st.markdown("The Precision-Recall curve highlights the balance between precision and recall for the Random Forest model.")