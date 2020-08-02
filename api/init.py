from flask import Flask
from flask_cors import CORS
# from database.models import *
import os


app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get('DATABASE_URL')
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db.init_app(app=app)


CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.after_request
def after_request(response):
    # response.headers.add( 'Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization,true')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    return response