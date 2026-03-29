@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   VisioALS — Build Installer
echo ============================================
echo.

:: Step 1: PyInstaller
echo [1/2] Building with PyInstaller...
call env\Scripts\activate.bat
pip install pyinstaller --quiet
pyinstaller --noconfirm VisioALS.spec
if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)
echo       Done — output in dist\VisioALS\
echo.

:: Step 2: Inno Setup
echo [2/2] Building installer with Inno Setup...

:: Try common install locations
set ISCC=
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
) else (
    echo WARNING: Inno Setup not found.
    echo Download it from https://jrsoftware.org/isdl.php
    echo Then run:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
    pause
    exit /b 1
)

"%ISCC%" installer.iss
if errorlevel 1 (
    echo ERROR: Inno Setup failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   SUCCESS!
echo   Installer: installer_output\VisioALS_Setup.exe
echo ============================================
pause
