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
import cv2
from PIL import Image
from numpy import asarray
import numpy as np

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
def apply_scale_factors(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

#load model function, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('water_bodies_segmentation_76.h5')
    return model

#image preprocessing function
def preprocess_image(img):
    size_x,size_y=128,128
    img = asarray(img) #convert image to array
    img = cv2.resize(img, (size_y, size_x))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)
    return img

########################## instructions ###############################
st.write('1. Select a region that is affected by floods on the map.\n2. Click "Export" button on the upper right corner of the map to download the coordinates of the region in a json file.\n3. Set the date range.\n4. Upload the json file that you downloaded in step 2. After uploading an RGB image of the selected region will be downloaded. \n5. Upload the image back to get the flood mapping of the region.')
##############################geemap####################################
m = leafmap.Map(center=(27.5570, 68.2028), zoom=10,
                locate_control=True, latlon_control=True, 
                draw_export=True, minimap_control=True)
m.add_basemap('ROADMAP')
m.to_streamlit(height=450)

############################date range sliders##########################
col1, col2, col3 = st.columns(3)
with col1:
    start_year = st.slider(
        'Start Year',
        min_value=2000,
        max_value=2022,
        step=1
    )

with col2:
    start_day = st.selectbox(
    'Start Day',
    (range(1,29,1)))
 
with col3:
    start_month = st.slider(
        'Start Month',
        min_value=1,
        max_value=12,
        step=1
    )

col4, col5, col6 = st.columns(3)
with col4:
    end_year = st.slider(
        'End Year',
        min_value=2000,
        max_value=2022,
        step=1
    )
with col5:
    end_day = st.selectbox(
    'End Day',
    (range(1,29,1)))

with col6:
    end_month = st.slider(
        'End Month',
        min_value=1,
        max_value=12,
        step=1
    )

start_year_converted = str(start_year)
start_day_converted = str(start_day)
start_month_converted = str(start_month)
end_year_converted = str(end_year)
end_day_converted = str(end_day)
end_month_converted = str(end_month)

start_date= start_year_converted+'-'+start_month_converted+'-'+start_day_converted
end_date= end_year_converted+'-'+end_month_converted+'-'+end_day_converted

col7,col8,col9 = st.columns(3)
with col7:
    st.write('Start Date: ', start_date)

with col9:
    st.write('End Date: ', end_date)

###################### json file uplaoder ###########################
# Create a file uploader widget
uploaded_file = st.file_uploader("Upload the GeoJSON file")

geometry=None

if uploaded_file is not None:
    ee.Initialize()
    # Open the JSON file
    geojson = json.loads(uploaded_file.read())

    # Parse the GeoJSON object
    parsed_json = json.loads(json.dumps(geojson))

    # Extract the coordinates from the GeoJSON object
    coordinates = parsed_json['features'][0]['geometry']['coordinates'][0]

    # Create an ee.Geometry object using the coordinates
    geometry = ee.Geometry.Polygon(coordinates)

    with st.spinner('Downloading Image...'):

        image_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry)

        median = image_collection.median()
        dataset = apply_scale_factors(median)
        out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
        filename = os.path.join(out_dir, 'image.tif')
        imageRGB = dataset.visualize(**{'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min':0.0, 'max': 0.3})
        geemap.ee_export_image(imageRGB,filename=filename,scale= 50, region=geometry)
    st.success('Image sotred in /Downloads directory')
else:
    st.write("Please upload a file.")

########################download image########################
# if st.button('Download Image'):
#     image_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
#         .filterDate(start_date, end_date) \
#         .filterBounds(geometry)

#     median = image_collection.median()
#     dataset = apply_scale_factors(median)
#     out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
#     filename = os.path.join(out_dir, 'image.tif')
#     imageRGB = dataset.visualize(**{'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min':0.0, 'max': 0.3})
#     geemap.ee_export_image(imageRGB,filename=filename,scale= 50, region=geometry)
#     st.write("Done!")

#########################

with st.spinner("Loading Model...."):
    model = load_model() #call load_model function to load the model

#ask user to upload image
file = st.file_uploader("Upload the downloaded image ", type=['png','jpg','jpeg','tif'])

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
    col1.image(original_img)
    col2.image(prediction_image)