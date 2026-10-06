@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call ".venv\Scripts\python.exe" -m pip install -e .
call ".venv\Scripts\python.exe" -m streamlit run app.py

