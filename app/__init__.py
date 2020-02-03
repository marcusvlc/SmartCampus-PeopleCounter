from flask import Flask
from app.ext import cors, socketio
from app.views import main

class App:

    _instance = None

    def create_app(config):
        ''' Create Flask app '''
        app = Flask(__name__)
        app.config.from_object(config)
        App.register_extensions(app)
        App.register_blueprints(app)
        App._instance = app
        return app

    def register_extensions(app):
        ''' Register app extensions '''
        cors.init_app(app)
        socketio.init_app(app)

    def register_blueprints(app):
        ''' Register Flask blueprints '''
        app.register_blueprint(main.blueprint)

    def app_context():
        return App._instance.app_context()