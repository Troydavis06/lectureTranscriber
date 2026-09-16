@echo off
rem Start the lecture notes daemon. Leave this window open while you watch.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m lecturetool %*
