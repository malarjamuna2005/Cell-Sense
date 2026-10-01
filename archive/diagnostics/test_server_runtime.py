import urllib.request
import json
import asyncio

try:
    import websockets
except ImportError:
    websockets = None

def main():
    print("=== TESTING ENDPOINTS ON RUNNING SERVER (localhost:8000) ===")
    
    # 1. Root /
    with urllib.request.urlopen("http://localhost:8000/", timeout=5) as r:
        html = r.read().decode("utf-8", errors="replace")
        print("GET / -> Status:", r.status)
        print("  Has loading-screen element:", 'id="loading-screen"' in html)
        print("  Has Connecting to EV Battery Gateway text:", "Connecting to EV Battery Gateway..." in html)
        print("  Has CELLSENSE title:", "CELLSENSE" in html)

    # 2. Docs /docs
    with urllib.request.urlopen("http://localhost:8000/docs", timeout=5) as r:
        print("GET /docs -> Status:", r.status)

    # 3. Health
    with urllib.request.urlopen("http://localhost:8000/api/health", timeout=5) as r:
        d = json.loads(r.read())
        print("GET /api/health ->", d)

    # 4. Device status
    with urllib.request.urlopen("http://localhost:8000/api/device/status", timeout=5) as r:
        d = json.loads(r.read())
        print("GET /api/device/status ->", d)

    # 5. Model info
    with urllib.request.urlopen("http://localhost:8000/api/ml/model", timeout=5) as r:
        d = json.loads(r.read())
        print("GET /api/ml/model -> Model:", d.get("model_type"), "| Status:", d.get("status"))

    # 6. WebSocket
    if websockets:
        async def test_ws():
            async with websockets.connect("ws://localhost:8000/ws/telemetry") as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                print("WS /ws/telemetry connected successfully! Received init msg:")
                print(" ", msg[:100], "...")
        asyncio.run(test_ws())

if __name__ == "__main__":
    main()
