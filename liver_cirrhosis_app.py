import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
bst = xgb.Booster()
bst.load_model('xgboost_model.json')

# Load the scaler and label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the Streamlit app
def main():
    st.set_page_config(
        page_title="Liver Disease Stage Prediction",
        page_icon=":hospital:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #001f3f; /* Light blue background */
            color: #ffffff; /* White text color */
        }
        .title {
            color: #0074D9; /* Lighter blue title */
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#001f3f, #001f3f);
            color: #ffffff;
        }
        .stButton>button {
            background-color: #0074D9; /* Button background color */
            color: white; /* Button text color */
            border: none;
            padding: 10px 20px;
            text-align: center;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 16px;
        }
        input {
            background-color: #001f3f; /* Input background color */
            color: white; /* Input text color */
        }
        .css-1b8uznv {
            color: white; /* Adjust dropdown color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image('liver_image2.png', use_column_width=True)

    st.title('Liver Disease Stage Prediction')

    st.write('Enter the patient details for prediction:')

    # User input for the features
    age = st.number_input('Age', min_value=0, max_value=100, value=51)
    sex = st.selectbox('Sex', options=['M', 'F'])
    ascites = st.selectbox('Ascites', options=['N', 'Y'])
    hepatomegaly = st.selectbox('Hepatomegaly', options=['N', 'Y'])
    spiders = st.selectbox('Spiders', options=['N', 'Y'])
    edema = st.selectbox('Edema', options=['N', 'Y'])
    bilirubin = st.number_input('Bilirubin', min_value=0.0, max_value=10.0, value=0.5)
    cholesterol = st.number_input('Cholesterol', min_value=0.0, max_value=500.0, value=149.0)
    albumin = st.number_input('Albumin', min_value=0.0, max_value=5.0, value=4.04)
    copper = st.number_input('Copper', min_value=0.0, max_value=500.0, value=227.0)
    alk_phos = st.number_input('Alk_Phos', min_value=0.0, max_value=1000.0, value=598.0)
    sgot = st.number_input('SGOT', min_value=0.0, max_value=100.0, value=52.70)
    triglycerides = st.number_input('Triglycerides', min_value=0.0, max_value=500.0, value=57.0)
    platelets = st.number_input('Platelets', min_value=0.0, max_value=1000.0, value=256.0)
    prothrombin = st.number_input('Prothrombin', min_value=0.0, max_value=20.0, value=9.9)

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Ascites': [ascites],
        'Hepatomegaly': [hepatomegaly],
        'Spiders': [spiders],
        'Edema': [edema],
        'Bilirubin': [bilirubin],
        'Cholesterol': [cholesterol],
        'Albumin': [albumin],
        'Copper': [copper],
        'Alk_Phos': [alk_phos],
        'SGOT': [sgot],
        'Tryglicerides': [triglycerides],
        'Platelets': [platelets],
        'Prothrombin': [prothrombin]
    })

    # Apply label encoding to categorical features
    input_data['Sex'] = input_data['Sex'].map({'M': 1, 'F': 0})
    input_data['Ascites'] = input_data['Ascites'].map({'N': 0, 'Y': 1})
    input_data['Hepatomegaly'] = input_data['Hepatomegaly'].map({'N': 0, 'Y': 1})
    input_data['Spiders'] = input_data['Spiders'].map({'N': 0, 'Y': 1})
    input_data['Edema'] = input_data['Edema'].map({'N': 0, 'Y': 1})

    # Standardize features using the same scaler
    input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    # Create DMatrix for the new instance
    dnew_instance = xgb.DMatrix(input_data)

    # Predict button
    if st.button('Predict'):
        # Predict on the new instance
        y_new_pred = bst.predict(dnew_instance)

        # Define the stage mapping
        stage_mapping = {0: 'Stage 1', 1: 'Stage 2', 2: 'Stage 3'}

        # Convert predictions to stages
        y_new_pred_labels = [stage_mapping[int(round(pred))] for pred in y_new_pred]

        
        # Output the prediction
        st.write(f'Predicted Stage for the New Patient is: {y_new_pred_labels[0]}')

if __name__ == "__main__":
    main()
