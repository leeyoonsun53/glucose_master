@echo off
echo ========================================
echo   혈당관리 마스터 - 자동 설치 스크립트
echo ========================================
echo.

REM 가상환경 존재 여부 확인
if not exist "venv" (
    echo [1/4] 가상환경 생성 중...
    python -m venv venv --without-pip
    if errorlevel 1 (
        echo Python이 설치되어 있지 않거나 PATH에 등록되지 않았습니다.
        pause
        exit /b 1
    )

    echo [2/4] pip 설치 중...
    venv\Scripts\python.exe -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
    venv\Scripts\python.exe get-pip.py pip==24.0
    del get-pip.py
) else (
    echo [1/4] 가상환경이 이미 존재합니다.
    echo [2/4] pip 확인 완료.
)

echo [3/4] 가상환경 활성화 중...
call venv\Scripts\activate.bat

echo [4/4] 필요한 패키지 설치 중...
pip install -r requirements.txt

echo.
echo ========================================
echo   설치 완료!
echo   실행하려면: run.bat
echo ========================================
pause
