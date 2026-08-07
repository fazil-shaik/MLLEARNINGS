import asyncio
import json
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from orchestrator import run_roundtable

app = FastAPI(title="The Roundtable")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        topic = websocket.query_params.get("topic", "")
        rounds = int(websocket.query_params.get("rounds", "2"))

        if not topic:
            await websocket.send_json({"type": "error", "text": "Topic is required"})
            await websocket.close()
            return

        result = await asyncio.to_thread(run_roundtable, topic, rounds, False)

        await websocket.send_json({"type": "round_start", "round": 1})
        for index, turn in enumerate(result["transcript"], start=1):
            await websocket.send_json({"type": "turn", "speaker": turn["speaker"], "text": turn["text"]})
            if index % 4 == 0 and index < len(result["transcript"]):
                await websocket.send_json({"type": "round_start", "round": (index // 4) + 1})

        await websocket.send_json({"type": "verdict", "text": result["verdict"]})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:  # pragma: no cover - defensive
        await websocket.send_json({"type": "error", "text": str(exc)})
        await websocket.close()
    finally:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
