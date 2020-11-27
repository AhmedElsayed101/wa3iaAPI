factors = {
        "q4": -0.01061726,
        "q5": 0.00294503, 
        "q6": -0.11403156, 
        "q7": -0.06913762, 
        "q8":  0.14710705, 
        "q9":  0.00142066 
    }
# answers = {
#     "question1": False,
#     "question2": False,
#     "question3": True,
#     "question4": 3,
#     "question5": 5,
#     "question6": 1,
#     "question7": 1,
#     "question8": 1,
#     "question9": 1
# }
def calculateRisk(answers):
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
