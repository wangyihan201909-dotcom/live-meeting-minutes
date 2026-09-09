"""网页控制台：把 engine.Engine 的事件推到浏览器。

    python app.py          然后浏览器会自动打开 http://127.0.0.1:8500

桌面版见 desktop.py，两者共用 engine.py 的编排逻辑。
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

import audio
import engine as eng
import meetings as mt
from engine import CONFIG_PATH, ROOT, load_config, save_config

CONSOLE_PORT = 8500


# ---------------------------------------------------------------- 事件总线

class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def broadcast(self, event: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def log(self, text: str, level: str = "info") -> None:
        await self.broadcast({
            "type": "log", "level": level,
            "ts": time.strftime("%H:%M:%S"), "text": text,
        })


APP = eng.Engine(Hub())
api = FastAPI()


class ConfigPatch(BaseModel):
    device: int | None = None
    loopback: bool | None = None
    auto_gain: bool | None = None
    n_cpu_moe: int | None = None
    min_chars: int | None = None
    max_wait: float | None = None
    max_pending: int | None = None


class TitlePatch(BaseModel):
    title: str = ""


@api.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@api.get("/api/state")
async def get_state() -> JSONResponse:
    try:
        devices = audio.list_devices()
    except Exception as exc:
        devices = []
        await APP.sink.log(f"枚举音频设备失败：{exc}", "error")
    return JSONResponse({
        "status": APP.status(),
        "config": APP.cfg,
        "devices": devices,
        "minutes": APP.minutes.to_json(),
        "transcript": {"lines": APP.transcript_lines, "buffer": APP.buffer_text},
    })


@api.post("/api/config")
async def patch_config(patch: ConfigPatch) -> JSONResponse:
    data = patch.model_dump(exclude_none=True)
    for key in ("device", "loopback", "auto_gain"):
        if key in data:
            APP.cfg["audio"][key] = data[key]
    if "n_cpu_moe" in data:
        APP.cfg["llm"]["n_cpu_moe"] = data["n_cpu_moe"]
    for key in ("min_chars", "max_wait", "max_pending"):
        if key in data:
            APP.cfg["summary"][key] = data[key]
    save_config(APP.cfg)
    await APP.sink.log("配置已保存" + ("（重启后生效）" if "n_cpu_moe" in data else ""))
    return JSONResponse(APP.cfg)


@api.post("/api/start")
async def start() -> JSONResponse:
    asyncio.create_task(APP.start())
    return JSONResponse({"ok": True})


@api.post("/api/stop")
async def stop() -> JSONResponse:
    await APP.stop()
    return JSONResponse({"ok": True})


# ---- 历史会议

@api.get("/api/meetings")
async def list_meetings(q: str = "") -> JSONResponse:
    return JSONResponse({"meetings": mt.listing(q)})


@api.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str) -> JSONResponse:
    m = mt.get(meeting_id)
    if m is None:
        return JSONResponse({"error": "没有这场会议"}, status_code=404)
    mt.repair_wav(m.audio_path)
    return JSONResponse({
        "meta": {**m.meta(), "has_audio": m.audio_path.exists()},
        "transcript": m.transcript(),
        "minutes": m.minutes_json(),
        "minutes_markdown": m.minutes_md(),
    })


@api.post("/api/meetings/{meeting_id}/title")
async def rename_meeting(meeting_id: str, patch: TitlePatch) -> JSONResponse:
    m = mt.get(meeting_id)
    if m is None:
        return JSONResponse({"error": "没有这场会议"}, status_code=404)
    title = patch.title.strip()
    if not title:
        return JSONResponse({"error": "标题不能为空"}, status_code=400)
    return JSONResponse(m.write_meta(title=title[:120]))


@api.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str) -> JSONResponse:
    if APP.meeting is not None and APP.meeting.id == meeting_id:
        return JSONResponse({"error": "这场会议正在进行，先停止再删"}, status_code=409)
    if not mt.remove(meeting_id):
        return JSONResponse({"error": "删除失败"}, status_code=404)
    return JSONResponse({"ok": True})


@api.get("/api/meetings/{meeting_id}/audio")
async def meeting_audio(meeting_id: str):
    m = mt.get(meeting_id)
    if m is None or not m.audio_path.exists():
        return JSONResponse({"error": "没有录音"}, status_code=404)
    mt.repair_wav(m.audio_path)          # 上次没正常收尾的话，头里长度是 0
    return FileResponse(m.audio_path, media_type="audio/wav")


@api.get("/api/meetings/{meeting_id}/export")
async def export_meeting(meeting_id: str, fmt: str = "md"):
    m = mt.get(meeting_id)
    if m is None:
        return JSONResponse({"error": "没有这场会议"}, status_code=404)
    title = m.meta().get("title") or m.id
    if fmt == "txt":
        body, mime, ext = mt.transcript_text(m), "text/plain", "转写.txt"
    elif fmt == "json":
        body = json.dumps(mt.export_bundle(m), ensure_ascii=False, indent=2)
        mime, ext = "application/json", "全部.json"
    elif fmt == "md":
        body, mime, ext = m.minutes_md(), "text/markdown", "纪要.md"
    else:
        return JSONResponse({"error": f"不支持的格式 {fmt}"}, status_code=400)
    # 标题是用户随手改的，冒号之类 Windows 不收，控制字符还会污染响应头
    safe = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "", title).strip(" .") or m.id
    name = quote(f"{safe[:80]}-{ext}")
    return Response(
        content=body.encode("utf-8"),
        media_type=f"{mime}; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{name}"},
    )


@api.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    APP.sink.clients.add(ws)
    await ws.send_json(APP.status())
    await ws.send_json({"type": "minutes", **APP.minutes.to_json()})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        APP.sink.clients.discard(ws)


@api.on_event("shutdown")
async def on_shutdown() -> None:
    await APP.stop()


def port_busy(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"缺少配置文件 {CONFIG_PATH}", file=sys.stderr)
        return 1
    if port_busy(CONSOLE_PORT):
        # 关掉黑窗口不一定会杀掉 python 子进程，重开就会撞端口
        print(f"\n端口 {CONSOLE_PORT} 已被占用，多半是上一次没退干净。", file=sys.stderr)
        print(f"控制台可能还开着：http://127.0.0.1:{CONSOLE_PORT}", file=sys.stderr)
        print("要强制结束旧进程，在 PowerShell 里执行：", file=sys.stderr)
        print(f'  Get-NetTCPConnection -LocalPort {CONSOLE_PORT} -State Listen | '
              'ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }', file=sys.stderr)
        return 1
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{CONSOLE_PORT}")).start()
    print(f"控制台：http://127.0.0.1:{CONSOLE_PORT}")
    uvicorn.run(api, host="127.0.0.1", port=CONSOLE_PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
