#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PORT=${PORT:-8000}
PYTHON=${PYTHON:-python3}
NGROK_URL=${NGROK_URL:-https://glutton-grab-squall.ngrok-free.dev}

if [ -x "$ROOT_DIR/env/bin/python" ]; then
  PYTHON="$ROOT_DIR/env/bin/python"
fi

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok is not installed. Install it from https://ngrok.com/download and run this script again." >&2
  exit 1
fi

if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  echo "A VisioALS web server is already running on port $PORT. Stop that older copy before relaunching so the latest code and credentials are used." >&2
  exit 1
fi

"$PYTHON" "$ROOT_DIR/web/server.py" --port "$PORT" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

attempt=0
until curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" || true
    echo "The VisioALS server exited during startup. Port $PORT may already be used by an older copy; stop it and run this launcher again." >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 30 ]; then
    echo "The VisioALS server did not start on port $PORT." >&2
    exit 1
  fi
  sleep 0.2
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  wait "$SERVER_PID" || true
  echo "The VisioALS server did not take ownership of port $PORT. Stop the older server and run this launcher again." >&2
  exit 1
fi

echo "Local demo ready at http://127.0.0.1:$PORT"
echo "Starting ngrok; press Ctrl+C to stop both services."
if [ -n "$NGROK_URL" ]; then
  echo "Using reserved public URL: $NGROK_URL"
  ngrok http "http://127.0.0.1:$PORT" --url "$NGROK_URL"
else
  echo "Share the HTTPS Forwarding URL shown below."
  ngrok http "http://127.0.0.1:$PORT"
fi
