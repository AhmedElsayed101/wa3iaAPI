"""
import pandas as pd 
import numpy as np
import pickle 

from statistics import mean, stdev
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')
df.dropna(axis=1 , inplace=True)
df.drop(labels=['id'] ,axis=1 , inplace= True)

label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

df = df.rename(columns= {'concave points_mean'  : 'concave_points_mean',
           				 'concave points_worst' : 'concave_points_worst'})

feature_selected_df = df[['diagnosis','texture_worst','radius_se','radius_worst','area_se',
             			  'area_worst','concave_points_mean','concave_points_worst']]


X = feature_selected_df.iloc[:,1:]
Y = feature_selected_df['diagnosis']

texture_worst_mean  = mean(X['texture_worst'])
texture_worst_stdev = stdev(X['texture_worst'])

radius_se_mean  = mean(X['radius_se'])
radius_se_stdev = stdev(X['radius_se'])

radius_worst_mean  = mean(X['radius_worst'])
radius_worst_stdev = stdev(X['radius_worst'])

area_se_mean  = mean(X['area_se'])
area_se_stdev = stdev(X['area_se'])

area_worst_mean  = mean(X['area_worst'])
area_worst_stdev = stdev(X['area_worst'])

concave_points_mean_mean  = mean(X['concave_points_mean'])
concave_points_mean_stdev = stdev(X['concave_points_mean'])

concave_points_worst_mean  = mean(X['concave_points_worst'])
concave_points_worst_stdev = stdev(X['concave_points_worst'])



std_sc = StandardScaler()
X_scale = std_sc.fit_transform(X)

x_train, x_val_test, y_train, y_val_test = train_test_split(X_scale, Y, test_size= 0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size= 0.5, random_state=42)

model = LogisticRegression(solver='lbfgs', random_state=42)
model.fit(x_train,y_train)


with open('model_pickle','wb') as f:
	pickle.dump(model,f)


with open('model_pickle','rb') as f:
	log_clf = pickle.load(f)


predictions = log_clf.predict(x_val)
acc = accuracy_score(y_val, predictions)


def model_prediction(model,input_data, texture_worst_mean, texture_worst_stdev, 
					 radius_se_mean, radius_se_stdev, radius_worst_mean,radius_worst_stdev,
					 area_se_mean,area_se_stdev,area_worst_mean,area_worst_stdev,
					 concave_points_mean_mean, concave_points_mean_stdev, concave_points_worst_mean, concave_points_worst_stdev):
	
	
	f1 = (input_data['texture_worst'] - texture_worst_mean) / texture_worst_stdev
	f2 = (input_data['radius_se'] - radius_se_mean) / radius_se_stdev
	f3 = (input_data['radius_worst'] - radius_worst_mean) / radius_worst_stdev
	f4 = (input_data['area_se'] - area_se_mean) / area_se_stdev
	f5 = (input_data['area_worst'] - area_worst_mean) / area_worst_stdev 
	f6 = (input_data['concave_points_mean'] - concave_points_mean_mean) / concave_points_mean_stdev
	f7 = (input_data['concave_points_worst'] - concave_points_worst_mean) / concave_points_worst_stdev

	f_list    = [f1,f2,f3,f4,f5,f6,f7]
	f_list_df = pd.DataFrame(np.array(f_list).reshape(1,7))

	prediction = model.predict(f_list_df)
	if prediction == 0 :
		print ('Benign.')
	else:
		print('Malignant.')

						
user_inpt = {'texture_worst' 		: 32.07000,
			 'radius_se'     		: 1.42500,
			 'radius_worst'  		: 13.39000,
			 'area_se'       		: 30.74000,
			 'area_worst'    		: 523.00000,
			 'concave_points_mean'  : 0.05246,
			 'concave_points_worst' : 0.23740
			 }

model_prediction(log_clf,user_inpt,texture_worst_mean,texture_worst_stdev,radius_se_mean, radius_se_stdev, radius_worst_mean,radius_worst_stdev,
	area_se_mean,area_se_stdev,area_worst_mean,area_worst_stdev,
	concave_points_mean_mean, concave_points_mean_stdev, concave_points_worst_mean, concave_points_worst_stdev)


"""
##################################################################################################################################################


import pandas as pd 
import numpy as np
import pickle 

from statistics import mean, stdev
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import load, dump


df = pd.read_csv('data.csv')
df.dropna(axis=1 , inplace=True)
df.drop(labels=['id'] ,axis=1 , inplace= True)

label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

df = df.rename(columns= {'concave points_mean'  : 'concave_points_mean',
           				 'concave points_worst' : 'concave_points_worst'})

feature_selected_df = df[['diagnosis','texture_worst','radius_se','radius_worst','area_se',
             			  'area_worst','concave_points_mean','concave_points_worst']]


X = feature_selected_df.iloc[:,1:]
Y = feature_selected_df['diagnosis']
std_sc = StandardScaler()
X_scale = std_sc.fit_transform(X)

dump(std_sc,'std_sc.bin', compress=True)

x_train, x_val_test, y_train, y_val_test = train_test_split(X_scale, Y, test_size= 0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size= 0.5, random_state=42)

model = LogisticRegression(solver='lbfgs', random_state=42)
model.fit(x_train,y_train)


with open('model_pickle','wb') as f:
	pickle.dump(model,f)


# with open('model_pickle','rb') as f:
# 	model = pickle.load(f)




# def model_prediction(model,user_input):
	
# 	std_sc = load('std_sc.bin')

# 	inputs = std_sc.transform(np.array(list(user_input.values())).reshape(1,7))
# 	prediction = model.predict(inputs)[0]

# 	if prediction == 0 :
# 		print ('Benign.')
# 	else:
# 		print('Malignant.')

						
# user_input = {'texture_worst' 		: 21.96,
# 			 'radius_se'     		: 0.1563,
# 			 'radius_worst'  		: 8.964,
# 			 'area_se'       		: 8.205,
# 			 'area_worst'    		: 242.2,
# 			 'concave_points_mean'  : 0.005917,
# 			 'concave_points_worst' : 0.02564
# 			 }

# model_prediction(model,user_input)


					