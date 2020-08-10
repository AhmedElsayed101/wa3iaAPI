# Wa3ia 

This wa3ia poject, it's an API for breast cancer detection app, you can calculate your risk after answering some qestions of receive a diagnosis, or predict your medical state using your mamogram.

All backend code follows [PEP8 style guidelines](https://www.python.org/dev/peps/pep-0008/). 

<hr/>

## Getting Started

### Project hierarchy

### Pre-requisites and Local Development 

Developers using this project should already have Python3, pip and postgresql installed on their local machines.

From the base directory run 
```
pip install requirements.txt
```
. All required packages are included in the requirements file. 

To run the application run the following commands: 
```
python app.py 
```
congratulations, server is running on (http://127.0.0.1:5000/)

<hr/>


## API Reference

### Getting Started

- Base URL:  This app can only be run locally on (http://127.0.0.1:5000/) 
  It is  hosted as a base URL on (https://wa3ia.herokuapp.com/)
- Authentication: This version of the application doesn't require any authentication.

<hr/>

### Error Handling
Errors are returned as JSON objects in the following format:
```
{
    "success": False, 
    "error": 400,
    "message": "bad request"
}
```
The API will return three error types when requests fail:
- 400: Bad Request
- 404: Resource Not Found
- 422: Not Processable 

<hr/>

### Endpoints 

### **POST**  '/api/risk'
#### - calculate your risk using some qestions

##### Payload
``` json
    {
        "question1": false,
        "question2": false,
        "question3": true,
        "question4": 3,
        "question5": 5,
        "question6": 1,
        "question7": 1,
        "question8": 1,
        "question9": 1
    }
```

##### Success Response
``` json

{
    "result" : "Your risk is 5.5151"
}

```
<hr/>

### **POST**  '/api/diagnosis'
#### - Use some questions to check if you are malignant or benign

##### Payload
``` json
    
{   
    "texture_worst" 		: 21.96,
    "radius_se"    		: 0.1563,
    "radius_worst"  		: 8.964,
    "area_se"       		: 9.205,
    "area_worst"    		: 242.2,
    "concave_points_mean"  : 0.005917,
    "concave_points_worst" : 0.02564
}
```

##### Success Response
``` json
    {

        "result" : "Malignant."
        
    }
```
<hr/>

### **POST**  '/api/prediction'
#### - use your mamogram to detect your case

##### Payload
``` json
{
    "base64Con" : "img encoded to base64"
}
```

##### Success Response
``` json
    
    {

        "prediction_output" : "normal",
        "confidence_output" : "51.5%"
        
    }
```

<hr/>


