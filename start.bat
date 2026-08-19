@echo off
chcp 65001 >nul
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"
pause
