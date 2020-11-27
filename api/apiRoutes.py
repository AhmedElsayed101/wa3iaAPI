from flask import jsonify, request
from init import app

from flask_cors import cross_origin


from utils.prediction import predict
@app.route('/api/prediction', methods=['POST'])
@cross_origin()
def predictMammogram():

    data = request.get_json()
    base64Con = data['base64Con']
    print(base64Con)
    result = predict(base64Con)    
    return jsonify(result)



from utils.diagnosis import model_prediction
@app.route('/api/diagnosis', methods=['POST'])
@cross_origin()
def diagnosis():
    
    data = request.get_json()         
    user_input = data
    result =  model_prediction(user_input)
    return jsonify({
        "result" : result
    })



from utils.risk import calculateRisk
@app.route('/api/risk', methods=['POST'])
@cross_origin()
def risk():
    
    data = request.get_json()
    answers = data
    result =  calculateRisk(answers)
    return jsonify({

        "result" : result
    })