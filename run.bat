@echo off
echo ========================================
echo   혈당관리 마스터 실행
echo ========================================

REM 현재 디렉토리를 스크립트 위치로 변경
cd /d "%~dp0"

REM Streamlit 앱 실행 (venv의 python 직접 사용)
venv\Scripts\python.exe -m streamlit run src/app.py

pause
