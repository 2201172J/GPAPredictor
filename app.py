import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('tunedGBR.pkl')
scaler = joblib.load('scaler.pkl')

# Define numerical columns
numericColumns = ['Age', 'StudyTimeWeekly', 'Absences']

# Define the prediction function
def predictGpa(data):
    featureNames = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
                     'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports',
                     'Music', 'Volunteering']
    
    # Ensure the input data has all the required features in the correct order
    data = data[featureNames]
    
    # Scale the numerical features
    data[numericColumns] = scaler.transform(data[numericColumns])
    
    prediction = model.predict(data)
    predictionClipped = np.clip(prediction, 0, 4.0)  # Clip the prediction to a maximum of 4.0
    return predictionClipped

# Streamlit web app
st.set_page_config(page_title = "GPA Prediction App", page_icon = ":books:", layout = "wide")

# Main header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Student GPA Predictor</h1>", unsafe_allow_html = True)
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Predict your GPA based on various features</h3>", unsafe_allow_html = True)
st.write("")

# Add custom CSS for better design
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
    }
    .block-container {
        max-width: 1200px;
        margin: auto;
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #e6f7ff;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html = True
)

# User inputs in the sidebar
st.sidebar.header("Input Features")
age = st.sidebar.number_input('Age', min_value = 15, max_value = 18)
gender = st.sidebar.selectbox('Gender', [0, 1], format_func = lambda x: 'Male' if x == 0 else 'Female')
ethnicity = st.sidebar.selectbox('Ethnicity', [0, 1, 2, 3], format_func = lambda x: ['Caucasian', 'African American', 'Asian', 'Other'][x])
parentalEducation = st.sidebar.selectbox('Parental Education', [0, 1, 2, 3, 4], format_func = lambda x: ['None', 'High School', 'Some College', 'Bachelor\'s', 'Higher'][x])
studyTimeWeekly = st.sidebar.number_input('Study Time Weekly', min_value = 0, max_value = 20)
absences = st.sidebar.number_input('Absences', min_value = 0, max_value = 30)
tutoring = st.sidebar.selectbox('Tutoring', [0, 1], format_func = lambda x: 'No' if x == 0 else 'Yes')
parentalSupport = st.sidebar.selectbox('Parental Support', [0, 1, 2, 3, 4], format_func = lambda x: ['None', 'Low', 'Moderate', 'High', 'Very High'][x])
extracurricular = st.sidebar.selectbox('Extracurricular', [0, 1], format_func = lambda x: 'No' if x == 0 else 'Yes')
sports = st.sidebar.selectbox('Sports', [0, 1], format_func = lambda x: 'No' if x == 0 else 'Yes')
music = st.sidebar.selectbox('Music', [0, 1], format_func = lambda x: 'No' if x == 0 else 'Yes')
volunteering = st.sidebar.selectbox('Volunteering', [0, 1], format_func = lambda x: 'No' if x == 0 else 'Yes')

# Create a DataFrame for the input data
inputData = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Ethnicity': [ethnicity],
    'ParentalEducation': [parentalEducation],
    'StudyTimeWeekly': [studyTimeWeekly],
    'Absences': [absences],
    'Tutoring': [tutoring],
    'ParentalSupport': [parentalSupport],
    'Extracurricular': [extracurricular],
    'Sports': [sports],
    'Music': [music],
    'Volunteering': [volunteering]
})

# Predict GPA
if st.sidebar.button('Predict GPA'):
    prediction = predictGpa(inputData)
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Predicted GPA</h2>", unsafe_allow_html = True)
    st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{prediction[0]:.2f}</h3>", unsafe_allow_html = True)
else:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Awaiting input features...</h2>", unsafe_allow_html = True)

# Add footer with styling
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #4CAF50;
            color: white;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
    </div>
""", unsafe_allow_html = True)
