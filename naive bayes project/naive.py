import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder

# Notebook & dataset info
notebooks_info = { 
    "Notebook 1 - GaussianNB": {
        "dataset": r"D:\Mishal\Semesters\5th Semester\NLP\NLP lab\naive bayes project\dataset\AnimalInformation.csv",
        "model_type": "gaussian"
    },
    "Notebook 2 - MultinomialNB": {
        "dataset": r"D:\Mishal\Semesters\5th Semester\NLP\NLP lab\naive bayes project\dataset\EmailSpamDetectionUpdated.csv",
        "model_type": "multinomial"
    },
    "Notebook 3 - BernoulliNB": {
        "dataset": r"D:\Mishal\Semesters\5th Semester\NLP\NLP lab\naive bayes project\dataset\LoanApprovalupdated.csv",
        "model_type": "bernoulli"
    },
    "Notebook 4 - Custom NB": {
        "dataset": r"D:\Mishal\Semesters\5th Semester\NLP\NLP lab\naive bayes project\dataset\weatherAndRoadCondition.csv",
        "model_type": "gaussian"  # assuming Gaussian for simplicity
    }
}

st.set_page_config(page_title="Naive Bayes Probability Calculator", layout="centered")
st.title("📊 Naive Bayes Probability Calculator")

# --- Step 1: Select dataset ---
selected_notebook = st.selectbox("Select Notebook / Dataset", list(notebooks_info.keys()))
dataset_path = notebooks_info[selected_notebook]["dataset"]
model_type = notebooks_info[selected_notebook]["model_type"]

# --- Step 2: Load dataset ---
df = pd.read_csv(dataset_path)
st.write("Preview of Dataset:")
st.dataframe(df.head())

# --- Step 3: Remove identifier/serial columns ---
# Skip columns named id, index, sr or numeric columns with all unique values
ignore_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'sr'] or df[col].nunique() == len(df)]
df = df.drop(columns=ignore_cols, errors='ignore')

st.write(f"Columns used for prediction: {list(df.columns)}")

# --- Step 4: Encode categorical features ---
le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # save encoder for later mapping

# --- Step 5: Dynamic input widgets ---
st.subheader("Select Conditions for Prediction")
features = df.columns[:-1]  # assuming last column is target
target = df.columns[-1]

conditions = {}
for feature in features:
    le = le_dict.get(feature)
    if le:
        conditions[feature] = st.selectbox(f"{feature}", le.classes_)
    else:
        min_val, max_val = int(df[feature].min()), int(df[feature].max())
        conditions[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=min_val)

# --- Step 6: Train model dynamically ---
X = df[features]
y = df[target]

if model_type == "gaussian":
    model = GaussianNB()
elif model_type == "multinomial":
    model = MultinomialNB()
elif model_type == "bernoulli":
    model = BernoulliNB()
else:
    st.error("Unknown model type!")
    st.stop()

model.fit(X, y)

# --- Step 7: Map inputs to numeric before prediction ---
input_dict = {}
for feature, value in conditions.items():
    le = le_dict.get(feature)
    if le:
        input_dict[feature] = int(le.transform([value])[0])
    else:
        input_dict[feature] = value

input_df = pd.DataFrame([input_dict])

# --- Step 8: Predict probability ---
if st.button("Calculate Probability"):
    prob = model.predict_proba(input_df)[0][1]  # probability of positive class
    st.success(f"✅ Predicted Probability: {prob}")


#GaussianNB – Works with continuous numeric features, assuming each feature follows a normal (Gaussian) distribution.
#MultinomialNB – Works with count-based features (integers ≥ 0), commonly used for text classification with word counts.
#BernoulliNB – Works with binary features (0/1), modeling presence or absence of an attribute, e.g., word occurrence in emails.
#Custom NB – User-defined Naive Bayes variant for special cases or mixed feature types, with custom probability calculations.


#py -m streamlit run "D:\Mishal\Semesters\5th Semester\NLP\NLP lab\naive bayes project\naive.py"
