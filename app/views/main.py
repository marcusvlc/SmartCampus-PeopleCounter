'''
Main App http endpoints
'''
from flask import Blueprint, abort, Response, jsonify, request
from flask_cors import cross_origin
from app.__init__ import App
from app.ext import socketio
import random
import datetime
from ..stream_detector import StreamDetector


blueprint = Blueprint('main', __name__, static_folder='static')


@blueprint.route("/404")
def deploy_test():
    return "Not found"

@blueprint.route("/couting/")
def people_couting_from_stream():
    url_video = request.args.get('url')

    sd = StreamDetector(url_video)

    return Response(sd.get_inference(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')