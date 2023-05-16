import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import platform

HEIGHT = 128
WIDTH = 128
CLASS_NAMES = ['lasagna', 'burrito', 'tacos', 'sushi', 'gyoza', 'pizza', 
               'sashimi', 'edamame', 'pasta', 'risotto', 'nachos', 'ramen']

if (platform.system().lower()=="linux"):
    bar = "/"
elif (platform.system().lower()=="windows"):
    bar = "\\"
else:
    bar = "/"

def read_data(path, im_size):
    X = []
    file_paths = []
    for file in os.listdir(path):
        image_path = os.path.join(path, file)
        file_paths.append(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, im_size)
        X.append(image)
    return np.array(X), np.array(file_paths)

def mapping(values, class_label_name):
    true_finalLabel = []
    for elem in values:
        true_finalLabel.append(class_label_name[elem])
    return true_finalLabel


def mapping_cuisine(values, class_label_name):
    italian_food = ["risotto", "pasta", "lasagna", "pizza"]
    japanese_food = ["gyoza", "sushi", "edamame", "sashimi", "ramen"] 
    mexican_food = ["tacos", "burrito", "nachos"]

    true_finalLabel = []
    for elem in values:
        if class_label_name[elem] in italian_food:
            true_finalLabel.append("italian")
        elif class_label_name[elem] in mexican_food:
            true_finalLabel.append("mexican")
        elif class_label_name[elem] in japanese_food:
            true_finalLabel.append("japanese")
        else:
            true_finalLabel.append("Unknown")

    return true_finalLabel







st.set_page_config(
    page_title = "My streamlit project",
    page_icon = "🍴",
    layout = "wide"

)

col1, col2, col3 = st.columns([1,5,1])
with col2:
    st.image(f"data{bar}CNN_finalModel.png")
    st.markdown("<h1 style='text-align: center; color: black;'>CNN classifier for food images</h1>", unsafe_allow_html=True)
    #st.title()

uploaded_file = st.sidebar.file_uploader("Upload your zip file with images", type=["zip"])
#OSS pandas funziona anche con un file direttamente e non con la string di dove sta il file
if uploaded_file is not None:
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall("images")
            folder_name = zip_ref.namelist()[0]
    except:
        st.write("error")

option = st.sidebar.selectbox("Select the view", ("Home", "Predictions"))


st.sidebar.write(option)


#with st.sidebar:
#    option = st.selectbox("Select the view", ("Home", "Visualization", "Map"))
#    st.write(option)
description = """
This app is part of the Machine Learning project I carried out at the Data Science bootcamp at The Bridge school.
I trained a convolutional neural network model (summarized in the banner picture of this website), to classify food 
pictures with the final aim of predicting the type of restaurant from pictures taken from TripAdvisor.
This is just a small project I carried out with limited time and computational resources, it could be improved a lot.
For instance one of the main improvement would be adding an additional CNN which filters food images from non-food images,
in order to give to the main CNN correct inputs. Moreover this model have been trained with only 12 food categories, 
corresponding to 3 types of cuisine. The categories and type of cuisines are the following:
- Risotto (Italian)
- Pasta (Italian)
- Lasagna (Italian)
- Pizza (Italian)
- Gyoza (Japanese)
- Sushi (Japanese)
- Sashimi (Japanese)
- Edamame (Japanese)
- Ramen (Japanese)
- Tacos (Mexican)
- Burrito (Mexican)
- Nachos  (Mexican)


I reached an accuracy on classifying these dishes of 0.65 (on validation set), and an accuracy on classifying type of cuisine
of 0.78.


Please, consider that uploading images of other food categories will not give good results.

"""
if option=="Home":
    st.subheader("Home")

    with st.expander("App details - click to show"):
        st.write(description)



elif option=="Predictions":
    if uploaded_file is not None:
        st.subheader("Predictions")

        class_label_name = {i:class_name for i ,class_name in enumerate(CLASS_NAMES)}
    
        X_test, file_paths = read_data(f"images{bar}{folder_name}", (HEIGHT,WIDTH))

        model = tf.keras.models.load_model(f"model{bar}model.h5")
        predictions = model.predict(tf.convert_to_tensor(X_test, dtype=tf.int32))
        preds_test = [x.argmax() for x in predictions]
        preds_names = mapping(preds_test, class_label_name)

        num_images = X_test.shape[0]
        num_cols = min(int(np.sqrt(len(X_test))), 5) 
        num_rows = num_images // num_cols + 1

        for i in range(num_rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < num_images:
                    cols[j].image(X_test[idx], use_column_width=True,
                                caption=f"{preds_names[idx]}")
                    
        preds_cuisine = mapping_cuisine(preds_test, class_label_name)

        final_prediction = max(set(preds_cuisine), key=preds_cuisine.count)
        st.markdown("### The CNN predicted that the type of cuisine is:")
        st.markdown(f"## {final_prediction}")
    else:
        st.markdown("#### Please upload a zip file containing the pictures for which you want to make predictions.")


                