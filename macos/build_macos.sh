#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This build script must run on macOS." >&2
    exit 1
fi

MACHINE_ARCH="$(uname -m)"
case "$MACHINE_ARCH" in
    arm64)
        TARGET_ARCH="arm64"
        DOWNLOAD_ARCH="Apple-Silicon"
        ;;
    x86_64)
        TARGET_ARCH="x86_64"
        DOWNLOAD_ARCH="Intel"
        ;;
    *)
        echo "Unsupported Mac architecture: $MACHINE_ARCH" >&2
        exit 1
        ;;
esac

RAW_VERSION="${VISIOALS_VERSION:-0.0.0}"
APP_VERSION="${RAW_VERSION#v}"
if [[ ! "$APP_VERSION" =~ ^[0-9]+([.][0-9]+){0,2}$ ]]; then
    echo "VISIOALS_VERSION must look like 1, 1.2, or 1.2.3 (optionally prefixed with v)." >&2
    exit 1
fi

BUILD_VENV="${VISIOALS_BUILD_VENV:-$PROJECT_ROOT/.build-venv-macos-$TARGET_ARCH}"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
WORK_DIR="$PROJECT_ROOT/build/macos-$TARGET_ARCH"
DIST_DIR="$PROJECT_ROOT/dist/macos-$TARGET_ARCH"
ICON_PATH="$WORK_DIR/VisioALS.icns"
WHEELHOUSE="$PROJECT_ROOT/build/wheelhouse-macos-$TARGET_ARCH"
RELEASE_DIR="$PROJECT_ROOT/release"
DMG_PATH="$RELEASE_DIR/VisioALS-$APP_VERSION-macOS-$DOWNLOAD_ARCH.dmg"

"$PYTHON_COMMAND" -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# Resolve against an explicit macOS 12 platform. On newer build machines pip
# otherwise prefers newer-OS wheel variants (for example SciPy's macOS 14
# build), which would make the finished app's stated minimum version untrue.
mkdir -p "$WHEELHOUSE"
"$BUILD_VENV/bin/python" -m pip download \
    --dest "$WHEELHOUSE" \
    --only-binary=:all: \
    --platform "macosx_12_0_$TARGET_ARCH" \
    --python-version 3.12 \
    --implementation cp \
    --abi cp312 \
    -r "$PROJECT_ROOT/requirements-macos.txt" \
    -r "$PROJECT_ROOT/requirements-build.txt"
"$BUILD_VENV/bin/python" -m pip install \
    --no-index \
    --find-links "$WHEELHOUSE" \
    -r "$PROJECT_ROOT/requirements-macos.txt" \
    -r "$PROJECT_ROOT/requirements-build.txt"

mkdir -p "$WORK_DIR" "$DIST_DIR" "$RELEASE_DIR"
"$BUILD_VENV/bin/python" "$SCRIPT_DIR/create_icon.py" "$ICON_PATH"

export VISIOALS_ICON="$ICON_PATH"
export VISIOALS_TARGET_ARCH="$TARGET_ARCH"
export VISIOALS_VERSION="$APP_VERSION"

"$BUILD_VENV/bin/pyinstaller" \
    --noconfirm \
    --clean \
    --workpath "$WORK_DIR/pyinstaller" \
    --distpath "$DIST_DIR" \
    "$PROJECT_ROOT/VisioALS.spec"

APP_PATH="$DIST_DIR/VisioALS.app"
if [[ ! -d "$APP_PATH" ]]; then
    echo "PyInstaller did not create $APP_PATH" >&2
    exit 1
fi

# PyInstaller ad-hoc signs the native libraries and final bundle when no paid
# signing identity is configured.
codesign --verify --deep --strict "$APP_PATH"

QT_QPA_PLATFORM=offscreen "$APP_PATH/Contents/MacOS/VisioALS" --smoke-test

DMG_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/visioals-dmg.XXXXXX")"
cleanup() {
    rm -rf "$DMG_TEMP"
}
trap cleanup EXIT

ditto "$APP_PATH" "$DMG_TEMP/VisioALS.app"
ln -s /Applications "$DMG_TEMP/Applications"
hdiutil create \
    -volname "VisioALS $APP_VERSION" \
    -srcfolder "$DMG_TEMP" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

echo "Created $DMG_PATH"
