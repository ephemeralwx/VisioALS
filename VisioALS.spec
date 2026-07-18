# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


is_macos = sys.platform == "darwin"
target_arch = os.environ.get("VISIOALS_TARGET_ARCH") or None
app_version = os.environ.get("VISIOALS_VERSION", "0.0.0").removeprefix("v")
icon_path = os.environ.get("VISIOALS_ICON")
if icon_path and not os.path.exists(icon_path):
    icon_path = None

datas = collect_data_files("mediapipe")
datas += collect_data_files("faster_whisper")
datas += collect_data_files("en_core_web_sm")
binaries = collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("ctranslate2")
hiddenimports = [
    "en_core_web_sm",
    "sklearn.svm",
    "sklearn.linear_model",
    "sklearn.neighbors",
    "sklearn.preprocessing",
    "sklearn.pipeline",
]

if not is_macos:
    hiddenimports.append("pyttsx3.drivers.sapi5")

excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "sentence_transformers",
    "tensorflow",
    "nltk",
    "sympy",
]
if is_macos:
    # VisioALS uses macOS' built-in `say` command for local fallback speech.
    # Excluding pyttsx3 avoids pulling the entire PyObjC framework collection
    # into the Mac download.
    excludes.extend(("pyttsx3", "pythoncom", "win32com"))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VisioALS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=not is_macos,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=not is_macos,
    upx_exclude=[],
    name="VisioALS",
)

if is_macos:
    app = BUNDLE(
        coll,
        name="VisioALS.app",
        icon=icon_path,
        bundle_identifier="org.visioals.desktop",
        version=app_version,
        info_plist={
            "CFBundleDisplayName": "VisioALS",
            "CFBundleName": "VisioALS",
            "LSApplicationCategoryType": "public.app-category.utilities",
            "LSMinimumSystemVersion": "12.0",
            "LSMultipleInstancesProhibited": True,
            "NSCameraUsageDescription": (
                "VisioALS uses the camera to track eye or head movement for "
                "hands-free response selection."
            ),
            "NSMicrophoneUsageDescription": (
                "VisioALS uses the microphone to hear and transcribe the "
                "caregiver's question and to record optional voice samples."
            ),
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
