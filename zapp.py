import pickle
import streamlit as st
import urllib.request
from sklearn.linear_model import LogisticRegression


# Raw URL to the model
url = "https://raw.githubusercontent.com/umeshpardeshi9545/Ecommerce-regression/blob/main/Ecom.pkl"

# Load the model
try:
    with urllib.request.urlopen(url) as response:
        model_data = response.read()  # Read bytes
        model = pickle.loads(model_data)  # Load model from bytes
except Exception as e:
    model = None
    st.error(f"Error loading the model: {e}")

# Streamlit UI
def main():
    st.title("E-Commerce Yearly Spending Prediction")

    # Input fields
    Avg_Session_Length = st.number_input("Avg Session Length (minutes)", min_value=0.0, format="%.2f")
    Time_on_App = st.number_input("Time on App (minutes)", min_value=0.0, format="%.2f")
    Time_on_Website = st.number_input("Time on Website (minutes)", min_value=0.0, format="%.2f")
    Length_of_Membership = st.number_input("Length of Membership (years)", min_value=0.0, format="%.2f")

    # Prediction button
    if st.button("Predict"):
        try:
            # Make prediction
            inputs = [[Avg_Session_Length, Time_on_App, Time_on_Website, Length_of_Membership]]
            prediction = model.predict(inputs)

            # Display result
            st.success(f"Predicted Yearly Amount Spent: ${prediction[0]:.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
