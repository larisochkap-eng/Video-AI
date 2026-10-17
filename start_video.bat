@echo off
cd /d "%~dp0"
py -3.12 video_gui_lowram.py
if errorlevel 1 pause
