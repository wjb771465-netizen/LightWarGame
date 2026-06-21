from flask import Flask


def create_app(web_ui):
    app = Flask(__name__)
    app.config["web_ui"] = web_ui
    from web.routes import register_routes
    register_routes(app)
    return app
