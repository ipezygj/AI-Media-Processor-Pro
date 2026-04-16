@echo off
title AI Media Processor Pro
cd /d "%~dp0"
python app_ui.py
if %errorlevel% neq 0 pause
