#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 VERSION" >&2
    echo "Example: $0 1.3.0" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

APP_VERSION="${1#v}"
if [[ ! "$APP_VERSION" =~ ^[0-9]+([.][0-9]+){2}$ ]]; then
    echo "VERSION must use MAJOR.MINOR.PATCH, for example 1.3.0." >&2
    exit 1
fi

cd "$PROJECT_ROOT"

if [[ ! -f ".github/workflows/macos-release.yml" ]]; then
    echo "The macOS GitHub Actions workflow is missing." >&2
    exit 1
fi

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
    echo "Check out a branch before publishing a release." >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit your changes before publishing a release." >&2
    echo "The GitHub builders can only package committed source code." >&2
    exit 1
fi

TAG="v$APP_VERSION"
if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
    echo "Tag $TAG already exists locally." >&2
    exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
    echo "Tag $TAG already exists on origin." >&2
    exit 1
fi

echo "Pushing $BRANCH to origin..."
git push origin "$BRANCH"

echo "Creating and pushing $TAG..."
git tag -a "$TAG" -m "VisioALS $TAG"
git push origin "$TAG"

REPOSITORY_URL="$(git remote get-url origin)"
REPOSITORY_URL="${REPOSITORY_URL%.git}"
if [[ "$REPOSITORY_URL" == git@github.com:* ]]; then
    REPOSITORY_URL="https://github.com/${REPOSITORY_URL#git@github.com:}"
fi

echo
echo "GitHub is now building the Apple Silicon and Intel DMGs."
echo "Actions: $REPOSITORY_URL/actions"
echo "Release: $REPOSITORY_URL/releases/tag/$TAG"
