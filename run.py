"""
Quick Launcher for Eraseon
Starts the local web application and opens it in your default browser.
"""

import os
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print("=" * 60)
    print(f"🚀 Launching Eraseon at {url}")
    print("=" * 60)

    # Open browser automatically after server boots
    def open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("backend.main:app", host=host, port=port, reload=False, log_level="info")
