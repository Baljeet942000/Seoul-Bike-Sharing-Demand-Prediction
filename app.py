import pickle
import datetime
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
!pip install xgboost
from xgboost import XGBRegressor

# loading xgboost model and standard scaler.
model = pickle.load(open('xgboost_regressor_r2_0_942_v1.pkl',"rb"))
sc = pickle.load(open('scaler_dump.pkl',"rb"))

# Creating a function to extract day,month and year from user input.
def get_string_to_datetime(date):
    dt = datetime.strptime(date, "%Y-%m-%d")
    return {"day": dt.day, "month": dt.month, "year": dt.year, "week_day": dt.strftime("%A")}

# Function for creating a seasons dataframe using user input of season.
def season_to_df(seasons):
    seasons_cols = ['Spring', 'Summer', 'Winter']
    seasons_data = np.zeros((1, len(seasons_cols)), dtype="int")

    df_seasons = pd.DataFrame(seasons_data, columns=seasons_cols)

    if seasons in seasons_cols:
        df_seasons[seasons] = 1
    return df_seasons

# Function for creating week-days dataframe using user input.
def days_df(week_day):
    days_names = ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    days_name_data = np.zeros((1, len(days_names)), dtype="int")
    df_days = pd.DataFrame(days_name_data, columns=days_names)

    if week_day in days_names:
        df_days[week_day] = 1
    return df_days

# Creating a streamlit application.
st.title("Bike Count Prediction")

date = str(st.date_input("Select a date",format = "DD/MM/YYYY"))
hour = st.number_input("Enter hour of the day (0-23)",min_value= 0,max_value= 23,step=1)
temperature = st.number_input("Temperature in degree celsius")
humiditity = st.number_input("Humidity in %",min_value= 0.0)
wind_speed = st.number_input("Wind_speed in m/s",min_value= 0.0)
visibility = st.number_input("visibility (10m)",min_value= 0.0)
solar_radiation = st.number_input("solar radiation in MJ/m2",min_value= 0.0)
rainfall = st.number_input("rainfall in mm: ",min_value= 0.0)
snowfall = st.number_input("snowfall in cm",min_value= 0.0)
seasons = st.selectbox("Season",['Autumn', 'Spring', 'Summer', 'Winter'])
holiday = st.selectbox("Is there a holiday today?",['Holiday','No Holiday'])
functioning_day = st.selectbox("Functioning day",['Yes','No'])

# Day,Month and Year dictionary.
str_to_date = get_string_to_datetime(date)

# Holiday dictionary(0/1)
holiday_dict = {"No Holiday": 0, "Holiday": 1}

# Functioning_day dictionary(0/1)
functioning_day_dict = {"No": 0, "Yes": 1}

# User input list.
user_input_list = [hour, temperature, humiditity, wind_speed, visibility, solar_radiation, rainfall, snowfall,
                   holiday_dict[holiday], functioning_day_dict[functioning_day], str_to_date['day'],
                   str_to_date["month"], str_to_date['year']]

# Features names list.
feature_names = ['Hour', 'Temperature(°C)', 'Humidity(%)',
                 'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)',
                 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day',
                 'Month', 'Year']

# Creating a dataframe with feature_names as columns and user_input_list as data.
df_user_input = pd.DataFrame([user_input_list], columns=feature_names)


# Merging df_user_input with seasons and week-days dataframe.
df_seasons = season_to_df(seasons)
df_days = days_df(str_to_date['week_day'])
df_for_pred = pd.concat([df_user_input, df_seasons, df_days], axis=1)

# Function for predicting bike-count.
def prediction(df):
    scaled_data = sc.transform(df)
    prediction = model.predict(scaled_data)

    return prediction

if st.button('Predict Count'):
    pred = round(prediction(df_for_pred)[0])

    if pred <= 0:
        st.subheader("Rented Bike count is : 0")
    else:
        st.subheader(f"Rented Bike count is : {pred}")




