import webbrowser
from web import create_app
from web.web_game_ui import WebGameUi

PORT = 5050

web_ui = WebGameUi()
app = create_app(web_ui)
try:
    webbrowser.open(f"http://localhost:{PORT}")
except Exception:
    pass
app.run(debug=True, threaded=True, port=PORT)
