import streamlit as st
import pickle
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split




st.title("PREDICTING LAPTOP PRICES THROUGH MACHINE LEARNING MODELS")

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))


# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input("weight of the laptop")

# Touchscreen
touchscreen = st.selectbox('TouchScreen', ['NO', 'Yes'])

# Touchscreen
ips = st.selectbox('IPS', ['NO', 'Yes'])


# Screen size
Screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', 
                                                '3200x1800','2880x1800', '2560x1600', '2560x1440', 
                                                '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# hardware
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# type of OS
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/Screen_size
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)

    query = query.reshape(1, 12)
     # Predict the price
    predicted_price = np.exp(pipe.predict(query))[0]

    # Display the original and predicted prices
    st.title("Predicted Laptop Price: ₹" + str(int(predicted_price)))
    
    filtered_df = df.loc[(df['Company'] == company) &
                         (df['TypeName'] == laptop_type) &
                         (df['Ram'] == ram) &
                         (df['Weight'] == weight) &
                         (df['Touchscreen'] == touchscreen) &
                         (df['IPS'] == ips) &
                         (df['Cpu brand'] == cpu) &
                         (df['HDD'] == hdd) &
                         (df['SSD'] == ssd) &
                         (df['Gpu brand'] == gpu) &
                         (df['os'] == os)]

    if not filtered_df.empty:
        original_price = filtered_df['Price'].values
        st.title("Original Laptop Price: ₹" + str(original_price[0]))
    
    else:
        st.title("Original Laptop Price: Not available")
    
    X=df.drop('Price',axis=1)
    print(X)
    y=df['Price']
    print(y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    y_pred1=np.exp(pipe.predict(X_test))
    print(y_pred1)
    print(y_test)
    df2=pd.DataFrame({"Actual":y_test[:51],"predicted":y_pred1[:51]})
    st.line_chart(df2,x=None,y=["Actual","predicted"],)
