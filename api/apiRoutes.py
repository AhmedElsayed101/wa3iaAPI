from flask import (render_template,
                   jsonify,
                   redirect, abort,
                   request
                   )
from app import app
# from database.models import *
# from .auth import requires_auth, token_required
import os


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
