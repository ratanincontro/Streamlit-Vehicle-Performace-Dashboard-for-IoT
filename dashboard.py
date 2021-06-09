import streamlit as st
import plotly_express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler 


uploaded_file = st.sidebar.file_uploader(label='Upload a CSV file',
				type=['csv'])


model = keras.models.load_model('HiPER.h5')
scaler = MinMaxScaler()


global df

if uploaded_file is None:
	st.title('No File Uploaded Yet')
	st.text('Please Upload a file from the sidebar')

if uploaded_file is not None:
	try:
		df = pd.read_csv(uploaded_file)
		if st.sidebar.checkbox('View Data'):
			st.title('Data')
			df
	except Exception as e:
		print(e)
		df = pd.read_excel(uploaded_file)

global numeric_columns
global x_values
if uploaded_file is not None:
	try:
		if st.sidebar.checkbox('View Map'):
			st.title('Route Map')
			map_data = [df['Latitude'],df['Longitude']]
			headers = ["latitude", "longitude"]
			df2 = pd.concat(map_data, axis=1, keys=headers)
			st.map(df2)
	except Exception as e:
		print(e)


#Widgets
if uploaded_file is not None:
	numeric_columns = list(df.select_dtypes(['float','int']).columns)
	datecol = df['Time']
	if st.sidebar.checkbox('View Chart'):
		st.title('Chart')
		slider_1, slider_2 = st.slider('Select Date Range' ,0,len(df)-1,[0,len(df)-1],1)
		col1, col2,col3,col4 = st.beta_columns(4)
		with col1: st.text('Start Date')
		with col2: st.text(df['Time'].iloc[slider_1])
		with col3: st.text('End Date')
		with col4: st.text(df['Time'].iloc[slider_2])
		if st.sidebar.checkbox('RPM'):
			st.write('RPM')
			st.line_chart(df['RPM'].iloc[slider_1:slider_2])
		if st.sidebar.checkbox('Speed'):
			st.write('Vehicle Speed')
			st.line_chart(df['Vehicle Speed'].iloc[slider_1:slider_2])
		if st.sidebar.checkbox('Fuel'):
			st.write('Fuel Rate (L-Hr)')
			st.line_chart(df['Fuel Rate (L-Hr)'].iloc[slider_1:slider_2])
		if st.sidebar.checkbox('Custom Chart'):
			kk = st.sidebar.selectbox('Custom Chart', options=numeric_columns)
			st.write(kk)
			start_date, end_date = df['Time'].iloc[slider_1], df['Time'].iloc[slider_1] 
			st.line_chart(df[kk].iloc[slider_1:slider_2])

if uploaded_file is not None:
	if st.sidebar.checkbox('Predict Torque'):
		st.title('Predict Inj Q Tor (mg-st)')
		rpm = st.number_input('Enter RPM')
		tps = st.number_input('Enter TPS(%)')
		if st.button('Predict'):
			New_Input = [[rpm,tps]]
			New_Input = scaler.transform(New_Input)
			predicted_torque = model.predict(New_Input)
			with col1: st.title('Predicted Inj Q Tor:')
			with col2: st.title(predicted_torque)


