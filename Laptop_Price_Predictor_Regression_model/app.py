import streamlit as st
import pickle
import numpy as np


# mporing model:
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
st.title("Laptop Predictor")

company = st.selectbox('Brand',df['Company'].unique())

# Type of laptop:
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB',[2,4,6,8,12,16,24,32,64])

# Weight:
weight = st.number_input('Weight of the Laptop')

# TouchScreen:
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# Screen Size:
screen_size = st.number_input('Screen Size')

# Resolution:
resolution = st.selectbox(
    'Screen Resolution',
    ['1024x768', '1280x720', '1366x768', '1440x900', '1600x900',
     '1920x1080', '1920x1200', '2560x1440', '2560x1600',
     '2880x1800', '3000x2000', '3200x1800', '3840x2160',
     '5120x2880', '6016x3384', '7680x4320']
)

# cpu:
cpu = st.selectbox('CPU Brand',df['Cpu brand'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# ssd

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# gpu:
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

# os
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
#  query:
    ppi = None
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
    ppi = ((x_res**2)+(y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    pipe.predict(query)
    st.title("The predicted price of this configuration is : "+str(int(np.exp(pipe.predict(query)[0]))))