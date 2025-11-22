import streamlit as st
import numpy as np
import pickle

# Load the trained KNN model
model = pickle.load(open('knn_model.pkl', 'rb'))

st.title("🚢 Titanic Survival Prediction (KNN Model)")
st.markdown("Predict whether a passenger survived or not.")

# Inputsas
# Take user input
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])


# Convert text values to numbers (same as used during training)
sex_map = {'male': 0, 'female': 1}
embarked_map = {'S': 0, 'C': 1, 'Q': 2}

# Create numeric input data for prediction
input_data = np.array([[pclass, sex_map[sex], age, sibsp, parch, fare, embarked_map[embarked]]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ The passenger would have survived!")
    else:
        st.error("❌ The passenger would not have survived.")