import streamlit as st
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt 
from src.predicter import Predicter
import json
import os
from PIL import Image

# streamlit run app.py
st.title('Heartbeat Classification App')

# Upload a mat file and show image
st.header('File uploading')

uploaded_files = st.file_uploader("Choose a .mat file", accept_multiple_files=True)
signal = None
for uploaded_file in uploaded_files:
    # print(uploaded_file)
    # print(type(uploaded_file))
    
    # mat = scipy.io.loadmat(uploaded_file)
    # signal = mat['data'][0]
    # st.write(signal.shape)
    
    bytes_data = uploaded_file.read()
    # st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)
    
    mat_file_path = os.path.join("samples", uploaded_file.name)
    
    st.header('Signal Visualization')
    mat = scipy.io.loadmat(mat_file_path)
    signal = mat['data'][0]
    # st.write(signal.shape)
    
# Draw image
# index = np.arange(0, 187, 1, dtype=int)
# plt.plot(index, signal, "-")
    signal_plot = signal.tolist()
    # print(signal_plot)
    plt.plot(signal_plot)
    
    signal_plot_path = os.path.join("media", "signal_plot.png")
    plt.savefig(signal_plot_path)
    
    image = Image.open(signal_plot_path)
    st.image(image, caption='Signal plot')
    
    st.header('Model Prediction')
    # Model prediction #???
    predicter = Predicter()
    
    # Predict the image in sample directory
    # os.chdir('..')
    predicter.PREDICT_SIGNAL_PATH = signal_plot_path
    prediction = predicter.predict()
    print(prediction)
    
    if prediction['label'] == "N":
        st.write("Normal beat")
        st.write("Confidence level: {}".format(round(prediction["prob_N"], 2)))
    elif prediction['label'] == "S":
        st.write("Supraventricular premature beat")
        st.write("Confidence level: {}".format(round(prediction["prob_S"], 2)))
    elif prediction['label'] == "P":
        st.write("Premature ventricular contraction")
        st.write("Confidence level: {}".format(round(prediction["prob_P"], 2)))
    elif prediction['label'] == "F":
        st.write("Fusion of ventricular and normal beat")
        st.write("Confidence level: {}".format(round(prediction["prob_F"], 2)))
    elif prediction['label'] == "U":
        st.write("Unclassifiable beat")
        st.write("Confidence level: {}".format(round(prediction["prob_U"], 2)))
    else:
        raise Exception("Label not defined")

