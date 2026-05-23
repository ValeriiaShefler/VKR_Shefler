import uvicorn
import webbrowser
import threading
import time
import sys
import os

# Добавляем путь к папке backend в sys.path
# Это нужно, чтобы Python нашёл модуль app.main
backend_path = os.path.join(os.path.dirname(__file__), "backend")
if os.path.exists(backend_path):
    sys.path.insert(0, backend_path)
else:
    # Если запускаем из собранного .exe, папка backend может быть внутри _internal
    internal_backend = os.path.join(sys._MEIPASS, "backend") if hasattr(sys, '_MEIPASS') else None
    if internal_backend and os.path.exists(internal_backend):
        sys.path.insert(0, internal_backend)

def open_browser():
    time.sleep(2)  # ждём, пока сервер поднимется
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    # Теперь импортируем app.main как модуль (без префикса backend, так как мы добавили backend в sys.path)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)