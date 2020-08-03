import pandas as pd 
import numpy as np
import pickle

from joblib import load, dump


with open('model_pickle','rb') as f:
	model = pickle.load(f)
	
def model_prediction(model,user_input):
	
	std_sc = load('std_sc.bin')

	inputs = std_sc.transform(np.array(list(user_input.values())).reshape(1,7))
	prediction = model.predict(inputs)[0]

	if prediction == 0 :
		return ('Benign.')
	else:
		return('Malignant.')

						
user_input = {'texture_worst' 		: 21.96,
			 'radius_se'     		: 0.1563,
			 'radius_worst'  		: 8.964,
			 'area_se'       		: 8.205,
			 'area_worst'    		: 242.2,
			 'concave_points_mean'  : 0.005917,
			 'concave_points_worst' : 0.02564
			 }

print ('result',model_prediction(model,user_input))