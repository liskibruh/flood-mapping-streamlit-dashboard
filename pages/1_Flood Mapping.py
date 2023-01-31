import streamlit as st
from streamlit_lottie import st_lottie
import leafmap.foliumap as leafmap
#leafmap.update_package()
import geojson
import json
import ee
import geemap
import os
import tensorflow as tf
from PIL import Image
from numpy import asarray
import numpy as np
import datetime
from skimage.morphology import opening
import skimage
from skimage import transform
from skimage import color

#title
st.set_page_config(page_title='Flood Detection', layout='wide')

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

st.title('This is the Flood Detection page')

#apply css from style.css
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#############################functions#################################
@st.cache(persist=True)
def ee_authenticate(token_name='EARTHENGINE_TOKEN'):
    geemap.ee_initialize(token_name=token_name)
#load model function, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('water_bodies_segmentation_76.h5')
    return model

#image preprocessing function
def preprocess_image(img):
    size_x,size_y=128,128
    img = asarray(img) #convert image to array
    img = transform.resize(img, (size_x, size_y))
    img = img[..., ::-1]
    img = np.expand_dims(img, axis=0)
    return img

#downlaod image function
def download_image():
    image_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)

    median = image_collection.median()
    dataset = apply_scale_factors(median)
    out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    filename = os.path.join(out_dir, 'image.tif')
    imageRGB = dataset.visualize(**{'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min':0.0, 'max': 0.3})
    geemap.ee_export_image(imageRGB,filename=filename,scale= 50, region=geometry)
    st.success('Image sotred in /Downloads')

#apply scaling factors to the downloaded image
def apply_scale_factors(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

#calculate precipitation function
@st.cache(allow_output_mutation=True)
def calculate_precipitation():
    with st.spinner('Calculating Precipitation...'):
        chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")
        filtered = chirps.filter(ee.Filter.date(start_date, end_date))
        total = filtered.reduce(ee.Reducer.sum()).clip(geometry) #Calculate rainfall over flood period
        precipitation_stats = geemap.image_stats(total, scale=5000) #Calculate average rainfall across a region
        precipitation_stats_values = precipitation_stats.getInfo()
        return precipitation_stats_values

########################## instructions ###############################
st.write('1. Select a region that is affected by floods on the map.\n2. Click "Export" button on the upper right corner of the map to download the coordinates of the region in a json file.\n3. Set the date range.\n4. Upload the json file that you downloaded in step 2. Click on Download Image button, an RGB image of the selected region will be downloaded. \n5. Upload the image back to get the flood mapping of the region.')

############################# main ####################################
ee_authenticate(token_name="EARTHENGINE_TOKEN")
row1_col1, row1_col2 = st.columns([2.5,1])
#Display Map
with row1_col1:
    m = leafmap.Map(center=(27.5570, 68.2028), zoom=10,
                    locate_control=True, latlon_control=True, 
                    draw_export=True, minimap_control=True)
    m.add_basemap('ROADMAP')
    m.to_streamlit(height=530)

#Date Widgets
with row1_col2: 
    start_date= st.date_input(
                label='Start Date',
                value=datetime.date(2022,1,1),
                min_value=datetime.date(2010,1,1),
                max_value=datetime.date(2022,12,30)
            )

    end_date= st.date_input(
                label='End Date',
                value=datetime.date(2022,1,1),
                min_value=datetime.date(2010,1,1),
                max_value=datetime.date(2022,12,30)
            )
    start_date=str(start_date)
    end_date=str(end_date)

    uploaded_file = st.file_uploader("Upload the GeoJSON file")
    geometry=None
    if uploaded_file is not None:
        #ee.Authenticate()
        geojson = json.loads(uploaded_file.read()) #open geojson file
        parsed_json = json.loads(json.dumps(geojson)) # Parse the GeoJSON object
        coordinates = parsed_json['features'][0]['geometry']['coordinates'][0] # Extract the coordinates from the GeoJSON object
        geometry = ee.Geometry.Polygon(coordinates) # Create an ee.Geometry object using the coordinates
        
        precipitation=calculate_precipitation() #calculate precipation for the region of interest
        st.metric(label = 'Precipitation', value = round(precipitation['mean']['precipitation_sum']))

    with st.spinner('Downloading Image...'):
        if st.button('Download Image'):
            download_image()

with st.spinner("Loading Model...."):
    model = load_model() #call load_model function to load the model

file = st.file_uploader("Upload the downloaded image ", type=['tif']) #ask user to upload image

if file is not None:
    image = Image.open(file) #open uploaded image
    prepd_img=preprocess_image(image) #preprocess image to meet the model's input requirements
    original_img= prepd_img.reshape((128,128,3))
    original_img = skimage.transform.resize(original_img, (400, 400))

    with st.spinner("Predicting..."):
        pred = model.predict(prepd_img) #pass the preprocessed image to the model to predict mask for it
        prediction_image = pred.reshape((128, 128, 1))
        prediction_image = skimage.transform.resize(prediction_image,(400,400))    
    #display images
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img)
        st.write('Original Image')
    with col2:
        st.image(prediction_image)
        st.write('Mapped Image')