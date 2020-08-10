from flask import (render_template,
                   jsonify,
                   redirect, abort,
                   request
                   )
from app import app

# from IPython.display import display  # Allows the use of display() for DataFrames

# import numpy as np
# import pickle
# import pandas as pd
# import os

import cv2
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input 
import numpy  as np 
import matplotlib.pyplot as plt
from PIL import Image 
import base64

# hbo lithy
import pandas as pd 
import numpy as np
import pickle

from joblib import load, dump


from .sayed_api import base64Con


@app.route('/api/prediction', methods=['POST'])
def predictPostCategory():

    data = request.get_json()

    print('data', data)

   
    base64Con = data['base64Con']

    
    
    # loading the model  
    model = tf.keras.models.load_model("transfer_224dim_3coulur_78acc.h5")

    # loading transfer learning layers 
    resnet_layers = tf.keras.models.load_model("resnet_layers_224dim_3coulur_78acc.h5")


    CATEGORIES = ["Benign", "InSitu","Invasive","Normal"]
    IMG_SIZE = 224



    # the code bellow would be changed every time we run 

    imgdata = base64.b64decode(base64Con)
    input_image = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(input_image , 'wb') as f:
        f.write(imgdata)


    #input of this method could be path of image or even the image 
    def prepare(input_image):
        #img_array = cv2.imread(filepath)
        img_array = cv2.imread(input_image)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE) )
        new_x = np.array(new_array).reshape(-3, IMG_SIZE, IMG_SIZE, 3)
        resnet_train_input = preprocess_input(new_x)
        x_train_features = resnet_layers.predict(resnet_train_input )
        return x_train_features

    

    prediction = model. predict_classes([prepare(input_image)])
    #np.max
    i=int (prediction)
    print("Predection :",(CATEGORIES[i]))

    prediction_output = CATEGORIES[i]


    prediction = model. predict([prepare(input_image)])
    confidence = 100*np.max(prediction)
    print("Confidence :",(round (confidence,2 ) ),"%")  

    confidence_output = (round (confidence,2 ) )



    return jsonify({

        "prediction_output" : prediction_output,
        "confidence_output" : confidence_output
        
    })



@app.route('/api/diagnosis', methods=['POST'])
def diagnosis():
    
    data = request.get_json()
    print('data', data)

    with open('model_pickle','rb') as f:
	    model = pickle.load(f)
	
    def model_prediction(model,user_input):
        
        
        std_sc = load('std_sc.bin')

        inputs = std_sc.transform(np.array(list(user_input.values())).reshape(1,7))
        prediction = model.predict(inputs)[0]

        if prediction == 0 :
            return ('Benign.')
        else:
            return ('Malignant.')

                       
    user_input = {'texture_worst' 		: 21.96,
                'radius_se'     		: 0.1563,
                'radius_worst'  		: 8.964,
                'area_se'       		: 8.205,
                'area_worst'    		: 242.2,
                'concave_points_mean'  : 0.005917,
                'concave_points_worst' : 0.02564
                }

    user_input = data

    result =  model_prediction(model,user_input)
    print('result', result)


    return jsonify({
        "result" : result
    })


@app.route('/api/risk', methods=['POST'])
def risk():
    
    data = request.get_json()

    
    factors = {
        "q4": -0.01061726,
        "q5": 0.00294503, 
        "q6": -0.11403156, 
        "q7": -0.06913762, 
        "q8":  0.14710705, 
        "q9":  0.00142066 
    }

    answers = {
        "question1": False,
        "question2": False,
        "question3": True,
        "question4": 3,
        "question5": 5,
        "question6": 1,
        "question7": 1,
        "question8": 1,
        "question9": 1
    }

    answers = data

    def CalculateRisk(answers, factors):
        if answers['question1'] == True:
            return "this tool doesn't work with women with medical history"
        elif answers['question2']  == True:
            return "this tool doesn't work with Women carrying a breast-cancer-producing mutation in BRCA1 or BRCA2"
        elif answers['question3'] == False:
            return "this tool doesn't work with women less than 35 years old"
        elif answers['question7']  == 0:
            Risk = abs((factors['q4'] * answers['question4'] ) + (factors['q5'] * answers['question5'] ) +(factors['q6'] * answers['question6'] )) * 100
        else:
            Risk = abs((factors['q4'] * answers['question4'] ) + (factors['q5'] * answers['question5'] ) +(factors['q6'] * answers['question6'] ) + (factors['q7'] * answers['question7'] ) + (factors['q8'] * answers['question8'] ) +(factors['q9'] * answers['question9'] )) * 100
    
        return f'Your risk is {round(Risk,4)}'

    result =  (CalculateRisk(answers, factors))

                
    
    return jsonify({

        "result" : result
    })