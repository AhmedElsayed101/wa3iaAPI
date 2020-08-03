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





@app.route('/api/prediction', methods=['POST'])
def predictPostCategory():

    data = request.get_json()

    # print(data)

   
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

    result =  model_prediction(model,user_input)
    print('result', result)

    return jsonify({

        "result" : result
    })





































# @app.route('/api')
# @token_required
# def api(payload):
#     return jsonify({
#         'message': 'Hello, Capstone!'
#     })


# @app.route('/login')
# def login():
#     audience = os.environ.get('API_AUDIENCE')
#     domain = os.environ.get('AUTH0_DOMAIN')
#     client_id = os.environ.get('CLIENT_ID')
#     redirect_url = os.environ.get('REDIRECT_URL')
    
#     part1 = f"https://{domain}/authorize?audience={audience}"
#     part2 = f"&response_type=token&client_id={client_id}"
#     part3 = f"&redirect_uri={redirect_url}"
#     url = part1+part2+part3

#     return redirect(url)

# @app.route('/api/actors/new')
# def add_actor_info():
#     # get users info from auth0 to store in database
#     return jsonify({
#         "actors":"all actors"
#     })


# @app.route('/logout')
# def logout():
#     return jsonify({
#         'message': 'You are logged out'
#     })


# @app.route('/api/actors')
# @requires_auth('view:actor')
# def get_all_actors(payload):

#     actros = Actor.query.order_by(Actor.id).all()

#     if len(actros) == 0:
#         abort(404)

#     actors_formatted = [actor.format() for actor in actros]

#     return jsonify({

#         "success": True,
#         "actors": actors_formatted,
#         "actors_number": len(actors_formatted)
#     })
