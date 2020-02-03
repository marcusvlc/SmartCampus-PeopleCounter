from app.__init__ import App

app = App.create_app('config')

SERVER_URL = app.config['SERVER_URL']
SERVER_PORT = app.config['SERVER_PORT']

if(__name__ == "__main__"):
    app.run(host=SERVER_URL, port=SERVER_PORT, debug=True, threaded=True)