import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

Base_Dir = Path(__file__).resolve().parent
model_path = Base_Dir/"House_Price_Prediction.pkl"
with open(model_path,'rb') as file:
    regressor = pickle.load(file)

# title
st.title('House Price Predection')

# input field
area = st.number_input('Area',min_value=0,value=100)
bedroom = st.number_input('Bedroom',min_value=0,value=3,step=1)
bathroom = st.number_input('Bathroom',min_value=0,value=2,step=1)
floor = st.number_input('Floors',min_value=0,value=1,step=1)
year_built= st.number_input('Year Built',min_value=1800,max_value=2026,value=2010,step=1)
condition = st.selectbox('Condition',['Poor','Fiar','Good','Excellent'])
garage = st.selectbox('Garage',['No','Yes'])
location = st.selectbox('Location',['Rural','Suburban','Urban'])

# convert condition to number
condition_map = {
    'Poor':0,
    'Fair':1,
    'Good':2,
    'Excellent':3
}

condition_encoded = condition_map[condition]

# convert garage to number

garage_map = {
    'No':0,
    'Yes':1
}

garage_encoded = garage_map[garage]

# create location  dummy variable
location_rural = 1 if location == 'Rural' else 0
location_suburban = 1 if location == 'Suburban' else 0
location_urban = 1 if location == 'Urban' else 0

# make prediction
if st.button('Predict Price'):
    input_data = pd.DataFrame({
        'Area' :[area],
        'Bedrooms':[bedroom],
        'Bathrooms':[bathroom],
        'Floors':[floor],
        'YearBuilt':[year_built],
        'Condition':[condition_encoded],
        'Garage':[garage_encoded],
        'Location_Rural':[location_rural],
        'Location_Suburban':[location_suburban],
        'Location_Urban':[location_urban]
    })

    prediction = regressor.predict(input_data)

    st.success(f"Predicted House Price:{prediction[0]:,.2f}")

