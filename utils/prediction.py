import cv2
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input 
import numpy  as np 
import matplotlib.pyplot as plt
from PIL import Image 
import base64


CATEGORIES = ["Benign", "InSitu","Invasive","Normal"]
IMG_SIZE = 224
OUT_PUT_IMAGE_PATH = 'models/images/some_image.jpg'  


def conv_base64_to_image(base64Con, output_image_path):
    imgdata = base64.b64decode(base64Con)
    input_image = output_image_path
    with open(input_image , 'wb') as f:
        f.write(imgdata)

def prepare(input_image, resnet_layers):
    #img_array = cv2.imread(filepath)
    img_array = cv2.imread(input_image)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE) )
    new_x = np.array(new_array).reshape(-3, IMG_SIZE, IMG_SIZE, 3)
    resnet_train_input = preprocess_input(new_x)
    x_train_features = resnet_layers.predict(resnet_train_input )
    return x_train_features

def predict(base64Con):
    
    # loading the model  
    model = tf.keras.models.load_model("models/transfer_224dim_3coulur_78acc.h5")

    # loading transfer learning layers 
    resnet_layers = tf.keras.models.load_model("models/resnet_layers_224dim_3coulur_78acc.h5")

    #input of this method could be path of image or even the image 
    conv_base64_to_image(base64Con, OUT_PUT_IMAGE_PATH)

    prediction = model.predict_classes([prepare(OUT_PUT_IMAGE_PATH, resnet_layers)])
    #np.max
    i = int (prediction)
    print("Predection :",(CATEGORIES[i]))
    prediction_output = CATEGORIES[i]

    prediction = model. predict([prepare(OUT_PUT_IMAGE_PATH, resnet_layers)])
    confidence = 100*np.max(prediction)
    print("Confidence :",(round (confidence,2 )),"%")  

    confidence_output = (round (confidence,2 ))

    return {
        "prediction_output" : prediction_output,
        "confidence_output" : confidence_output
    }

