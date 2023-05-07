import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image as image_utils
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.utils import pad_sequences
from keras import models

# Load images from the saved pickle file
with open('images.pickle', 'rb') as handle:
    images = pickle.load(handle)

df_txt0 = pd.read_csv('df2.csv')

dimages, keepindex = [], []
# Creating a dataframe where only the first caption is taken for processing
df_txt0 = df_txt0[df_txt0['index'] == 0]
for i, fnm in enumerate(df_txt0.filename):
    if fnm in images.keys():
        dimages.append(images[fnm])
        keepindex.append(i)

# fnames are the names of the image files
fnames = df_txt0['filename'].iloc[keepindex].values
# dcaptions are the captions of the images
dcaptions = df_txt0['caption'].iloc[keepindex].values
# dimages are the actual features of the images
dimages = np.array(dimages)

# The maximum number of words in the dictionary
nb_words = 6000
tokenizer = Tokenizer(num_words=nb_words)
tokenizer.fit_on_texts(dcaptions)
vocab_size = len(tokenizer.word_index) + 1
print(f"vocabulary size: {vocab_size}")
dtexts = tokenizer.texts_to_sequences(dcaptions)

modelvgg = VGG16(include_top=True, weights=None)
# Load the locally saved weights
modelvgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5")
modelvgg.layers.pop()
modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)

# Loading the model
model1 = load_model('caption_model.h5')

index_word = {index: word for word, index in tokenizer.word_index.items()}

def predict_caption(image):
    """
    image.shape = (1, 224, 224, 3)
    """
    in_text = 'startseq'
    maxlen = 30

    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=maxlen)
        yhat = model1.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word[yhat]
        in_text += " " + newword
        if newword == "endseq":
            break
    text = in_text.split(' ')
    cap = ''
    for i in range(1, len(text) - 1):
        cap += text[i] + ' '

    return cap

# Streamlit app code
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Image Captioning App")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpeg","jpg"])

npix = 224  # Image size is fixed at 224 because VGG16 model has been pre-trained to take that size.
target_size = (npix, npix, 3)

if uploaded_file is not None:
    # Preprocess the image and generate the caption
    image = load_img(uploaded_file, target_size=target_size)
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    nimage = preprocess_input(image)
    y_pred = modelvgg.predict(nimage.reshape((1,) + nimage.shape[:3]))
    ypred = y_pred.flatten()
    dimage = np.array(ypred)

    # Generate the caption for the image
    caption = predict_caption(dimage.reshape(1, len(dimage)))

    # Display the image and the caption
    st.image(uploaded_file, caption="Uploaded Image")
    st.write("Caption: ", caption, "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)
    st.write("<style>div.row-widget.stRadio > div{flex-direction:row;}</style>", unsafe_allow_html=True)

