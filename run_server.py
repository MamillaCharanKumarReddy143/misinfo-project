from app import app
import os
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new("http://127.0.0.1:3000/")

if __name__ == "__main__":
    # Wait 1.5 seconds for the server to start, then open Chrome/default browser
    Timer(1.5, open_browser).start()
    
    # Ensure port 3000 is used
    app.run(host='0.0.0.0', port=3000, debug=True)
