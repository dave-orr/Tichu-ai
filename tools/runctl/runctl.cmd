@echo off
rem runctl from any shell: runctl status, runctl pause dmc1, ... (see runctl.py)
"%USERPROFILE%\projects\tichu-ai\.venv\Scripts\python.exe" "%~dp0runctl.py" %*
