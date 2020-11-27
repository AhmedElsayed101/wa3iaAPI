import numpy as np
import pickle
from joblib import load, dump


with open('models/model_pickle','rb') as f:
    model = pickle.load(f)

def model_prediction(user_input):
    
    
    std_sc = load('models/std_sc.bin')

    inputs = std_sc.transform(np.array(list(user_input.values())).reshape(1,7))
    prediction = model.predict(inputs)[0]

    if prediction == 0 :
        return ('Benign.')
    else:
        return ('Malignant.')
