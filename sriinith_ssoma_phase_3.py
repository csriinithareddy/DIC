import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc


def main():
    st.title("Airline Passenger Satisfaction Prediction")
    st.write("This app predicts the passenger satisfaction of an airline using the Random Forest Classifier")

    # Load the data
    train_df, test_df = load_data()

    # Preprocessing the data
    df = preprocess_data(train_df, test_df)
    
    # Train model
    model,feature_names = train_model(df)
    
    # Get feature importances
    feature_importances = model.feature_importances_

    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort DataFrame by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    st.sidebar.header('User Input Parameters')
    
    with st.sidebar.form(key="user_input_form"):
        gender = st.selectbox('gender', ['Female', 'Male'])
        customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'Disloyal Customer'])
        age = st.slider('Age', int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
        type_of_travel = st.selectbox('Type of Travel', ['Personal Travel', 'Business travel'])
        flight_class = st.selectbox('Class',['Eco Plus', 'Business','Eco'])
        flight_distance = st.slider('Flight Distance (in minutes)', int(df['Flight Distance'].min()), int(df['Flight Distance'].max()), int(df['Flight Distance'].mean()))
        inflight_wifi_service = st.slider('Inflight wifi service', 0, 5, 3)
        departure_arrival_time_convenient = st.slider('Departure/Arrival time convenient', 0, 5, 2)
        ease_of_online_booking = st.slider('Ease of online booking', 0, 5, 4)
        gate_location = st.slider('Gate location', 1, 5, 3)
        food_and_drink = st.slider('Food and drink', 0, 5, 4)
        online_boarding = st.slider('Online boarding', 0, 5, 2)
        seat_comfort = st.slider('Seat comfort', 0, 5, 3)
        inflight_entertainment = st.slider('Inflight entertainment', 0, 5, 3)
        onboard_service = st.slider('Onboard service', 1, 5, 2)
        leg_room_service = st.slider('Leg room service', 0, 5, 3)
        baggage_handling = st.slider('Baggage handling', 1, 5, 4)
        checkin_service = st.slider('Checkin service', 0, 5, 3)
        inflight_service = st.slider('Inflight service', 0, 5, 2)
        cleanliness = st.slider('Cleanliness', 0, 5, 3)
        arrival_delay_minutes = st.slider('Arrival delay in minutes', 0, int(df['Arrival Delay in Minutes'].max()), int(df['Arrival Delay in Minutes'].mean()))

        predict_button = st.form_submit_button("Predict")
    url = "https://www.washingtonpost.com/wp-apps/imrs.php?src=https://arc-anglerfish-washpost-prod-washpost.s3.amazonaws.com/public/OAF4JDXUT4I6XJRWDDFMLGUY3Q.jpg&w=916"  # Replace with the URL of your animation or image
    st.image(url, use_column_width=True)
    
    if predict_button:
        user_input = get_user_input(df, gender, customer_type, age, type_of_travel, flight_class, flight_distance,
                                    inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking,
                                    gate_location, food_and_drink, online_boarding, seat_comfort,
                                    inflight_entertainment, onboard_service, leg_room_service, baggage_handling,
                                    checkin_service, inflight_service, cleanliness, arrival_delay_minutes)

        preprocessed_input = preprocess_input(user_input)

        satisfaction_pred = model.predict(preprocessed_input)[0]
        #st.subheader('Satisfaction Prediction:')
        #st.write(satisfaction_pred)
        if satisfaction_pred == "neutral or dissatisfied":
            satisfaction_label = "<span style='color: red'>Dissatisfied</span>"
        else:
            satisfaction_label = "<span style='color: green'>Satisfied</span>"

        satisfaction_html = f"<h2 style='font-size: 24px;'>Customer Satisfaction: {satisfaction_label}</h2>"
        st.markdown(satisfaction_html, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.pie(importance_df['Importance'], labels=importance_df['Feature'], autopct='%1.1f%%')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
        
    # Add animation or image from the internet
    #st.subheader('Animation or Image from the Internet')
   
        
# Apparently the steps performed inside this are performed in the function preprocess_data too.
def preprocess_input(data):
    
    # dropping the unecessary column
    df_input = pd.DataFrame(data, index=[0])
    
    #performing label encoding on the columns that have categorical features
    le = LabelEncoder()
    cat_col = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for i in cat_col:
        df_input[i] = le.fit_transform(df_input[i])
    return df_input.values

def load_data():
    air_train_df,air_test_df = pd.read_csv("train.csv"),pd.read_csv("test.csv")
    return air_train_df, air_test_df

def preprocess_data(train_df, test_df):
    
    #concatenating train and test dataframes
    f_df = pd.concat([train_df, test_df])
    
    #dropping unecesary columns 
    f_df = f_df.drop(['Unnamed: 0', 'id'], axis=1)
    
    #filling the missing values in the column "arrival delay in minutes" with 0's
    f_df['Arrival Delay in Minutes'].fillna(0, inplace=True)
    
    #dropping the column "departure delay in minutes' due to it's high correlation with the column "arrival delay"
    f_df = f_df.drop('Departure Delay in Minutes', axis=1)
    
    
    nod_sat_df = f_df
    nod_df = f_df
    nod_df = nod_df[nod_df['satisfaction'] == "neutral or dissatisfied"]
    nod_df = nod_df.drop(nod_df.index[:18000])
    nod_sat_df = nod_sat_df.drop(nod_sat_df[nod_sat_df['satisfaction'] == 'neutral or dissatisfied'].index)
    df = pd.concat([nod_sat_df, nod_df])

    #performing label encoding on the categorical columns in the dataset
    le = LabelEncoder()
    cat_col = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for i in cat_col:
        df[i] = le.fit_transform(df[i])
    #gender_mapping = {0: 'Female', 1: 'Male'}
    #df['Gender'] = df['Gender'].map(gender_mapping)
    
     

    return df

def train_model(df):
    
    # Splitting the data into train and test - before that we drop the target column from the dataset to form the train dataset
    X = df.drop(['satisfaction'], axis=1)
    y = df['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)
    return random_forest_model,X_train.columns

def get_user_input(df, gender, customer_type, age, type_of_travel, flight_class, flight_distance,
                  inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking,
                  gate_location, food_and_drink, online_boarding, seat_comfort,
                  inflight_entertainment, onboard_service, leg_room_service, baggage_handling,
                  checkin_service, inflight_service, cleanliness, arrival_delay_minutes):

    data = {
        'Gender': gender,
        'Customer Type': customer_type,
        'Age': age,
        'Type of Travel': type_of_travel,
        'Class': flight_class,
        'Flight Distance': flight_distance,
        'Inflight wifi service': inflight_wifi_service,
        'Departure/Arrival time convenient': departure_arrival_time_convenient,
        'Ease of Online booking': ease_of_online_booking,
        'Gate location': gate_location,
        'Food and drink': food_and_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': inflight_entertainment,
        'On-board service': onboard_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Arrival Delay in Minutes': arrival_delay_minutes
    }

    return data


if __name__ == "__main__":
    main()


