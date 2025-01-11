import streamlit as st
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

example1 = [4,             # guests
            1,             # beds
            1,             # bedrooms
            1,             # bathrooms
            2026,          # year
            9,             # month
            1,             # kitchen
            0,             # free_parking
            1,             # security camera
            1,             # elevator
            3,             # experience host
            1,             # TV
            0,             # laundry
            0,             # air conditioning
            0,             # dryer
            0,             # swimming pool
            21.06914,      # lat
            105.81192]     # lon
example1_results = 1271667


# Load the model
@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    print("Model loaded successfully.")
    return loaded_model


model = load_model()

options = {}
# Preprocessing function
def preprocessing(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


# Get latitude and longitude
def get_lat_lon(city, district):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(f"{district}, {city}")
    if location:
        return location.latitude, location.longitude
    else:
        return None, None


# Streamlit app
st.title("Hotel Price Prediction in Hanoi")

st.write("Enter location details to fetch latitude and longitude:")

district = st.text_input("Enter District:", "Hoan Kiem")

lat, lon = None, None
if st.button("Fetch Latitude and Longitude"):
    lat, lon = get_lat_lon("Hanoi", district)
    if lat and lon:
        st.success(f"Latitude: {lat}, Longitude: {lon}")
        options['lat'] = lat
        options['lon'] = lon
    else:
        st.error("Could not fetch latitude and longitude. Please check the inputs.")

# Define columns
columns_num = ['guests', 'beds', 'bedrooms', 'bathrooms','host experience', 'year', 'month']
columns_check = ['kitchen', 'free_parking', 'security camera', 'elevator',
                 'TV', 'laundry', 'air conditioning',
                 'dryer', 'swimming pool']
columns = columns_num + columns_check + ['lat', 'lon']

# Create a form

with st.form("feature_form"):
    st.write("Numeric features:")
    for col in columns_num:
        options[col] = st.number_input(f"Enter value for {col}",
                                       value=2025 if col == 'year' else 3 if col == 'month' else 4 if col == 'guests' else 2 if col == 'beds' else 1 if col == 'bedrooms' else 1 if col == 'bathrooms' else 4,
                                       step=1, format="%d")

    st.write("Has features (check for yes):")
    for col in columns_check:
        options[col] = st.checkbox(f"{col}", value=False)

    lat, lon = get_lat_lon("Hanoi", district)
    options['lat'] = lat
    options['lon'] = lon

    submitted = st.form_submit_button("Submit")

# Handle submission
if submitted:
    # Prepare input data
    try:
        # Check for invalid inputs
        invalid_inputs = []
        for col in columns_num:
            if options[col] <= 0:
                invalid_inputs.append(col)

        if invalid_inputs:
            st.error(f"The following fields must have values greater than 0: {', '.join(invalid_inputs)}")
        else:
            input_data = np.array([
                options[col] if col in columns_num + ['lat', 'lon']
                else 1 if options[col] else 0 for col in columns
            ])
            input_data = input_data.reshape(1, -1)

            # Display selections
            st.write("Your selections:")
            st.write(dict(zip(columns, input_data.flatten())))

            # Make prediction
            prediction = model.predict(input_data)
            st.write("Prediction of price:")
            st.write(prediction[0])

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write('Try our example: ')
with st.form("example1_form"):
    example1_button = st.form_submit_button("Example1")
if example1_button:
    st.write(dict(zip(columns, example1)))
    input_data = np.array(example1).reshape(1, -1)

    prediction = model.predict(input_data)
    st.write("Prediction of price:")
    st.write(prediction[0])
    st.write("Real Price: ", example1_results)


