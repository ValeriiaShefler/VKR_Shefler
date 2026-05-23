chcp 65001 >nul
@echo off
setlocal
set PYTHONHOME=%~dp0python
set PYTHONPATH=%~dp0python\Lib\site-packages

start /B %~dp0python\python.exe run_server.py
timeout /t 3 /nobreak >nul

echo Сервер запущен. Закройте это окно и окно браузера для остановки.
pause >nul

:: Остановка сервера при выходе
taskkill /f /im python.exe >nul 2>&1