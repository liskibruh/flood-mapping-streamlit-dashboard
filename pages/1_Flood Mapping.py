import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from numpy import asarray
import numpy as np
import json
from streamlit_lottie import st_lottie

st.set_page_config(page_title='Flood Mapping', layout='wide')

#add lottie animation
def load_lottiefile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

lottie_file1 = load_lottiefile('63605-flood.json')
lottie_file2 = load_lottiefile('74775-satellite-around-earth.json')
col1, col2, col3= st.columns(3)
with col1:
    st_lottie(
    lottie_file1,
    reverse=False,
    loop=True,
    quality='low',
    height=100,
    width=200
)
with col3:
    st_lottie(
    lottie_file2,
    reverse=False,
    loop=True,
    quality='low',
    height=100,
    width=200
)

#title
st.title('Flood Water Mapping')

#apply css from style.css
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#load model function, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('water_bodies_segmentation_76.h5')
    return model

with st.spinner("Loading Model...."):
    model = load_model() #call load_model function to load the model

#image preprocessing function
def preprocess_image(img):
    size_x,size_y=128,128
    img = asarray(img) #convert image to array
    img = cv2.resize(img, (size_y, size_x))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)
    return img

#ask user to upload image
file = st.file_uploader("Upload image ", type=['png','jpg','jpeg'])

if file is not None:
    image = Image.open(file) #open uploaded image

    prepd_img=preprocess_image(image) #preprocess image to meet the model's input requirements
    original_img= prepd_img.reshape((128,128,3))
    original_img= cv2.resize(original_img,(400,400))

    with st.spinner("Predicting..."):
        pred = model.predict(prepd_img) #pass the preprocessed image to the model to predict mask for it
        prediction_image = pred.reshape((128, 128, 1))
        prediction_image = cv2.resize(prediction_image,(400,400)) #resize the predicted mask image so it's displayed bigger on the web app page
    
    #display images
    col1, col2 = st.columns(2)
    #col1.image(prepd_img)
    col1.image(original_img)
    col2.image(prediction_image)
