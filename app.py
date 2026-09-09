"""一键启动器：拉起转写服务 + 摘要模型，采集音频，输出滚动纪要，并提供网页控制台。

    python app.py          然后浏览器会自动打开 http://127.0.0.1:8500

三个进程由本程序统一管理：
    llama-server   摘要模型（OpenAI 兼容接口）
    wlk            WhisperLiveKit 转写服务
    本进程          采音频 -> 推转写 -> 抽取合并 -> 写纪要 -> 推网页
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

import audio
import meetings as mt
import minutes as mn

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
CONSOLE_PORT = 8500


# ---------------------------------------------------------------- 配置

def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def save_config(cfg: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
        fh.write("\n")          # 少了末尾换行，页面上每改一次配置就多一条无谓的 git diff
    os.replace(tmp, CONFIG_PATH)


def llm_cmd(cfg: dict) -> list[str]:
    c = cfg["llm"]
    return [
        c["binary"], "-m", c["model_path"],
        "--n-cpu-moe", str(c["n_cpu_moe"]),
        "-c", str(c["ctx"]), "-ngl", "99",
        "--host", "127.0.0.1", "--port", str(c["port"]),
    ] + list(c.get("extra_args", []))


def asr_cmd(cfg: dict) -> list[str]:
    c = cfg["asr"]
    cmd = [
        c["binary"], "--backend", c["backend"], "--language", c["language"],
        "--pcm-input", "--host", "127.0.0.1", "--port", str(c["port"]),
    ]
    if c.get("model_path"):
        # 指向本地目录，启动时就不会去连 HuggingFace（实测 hf-mirror 也连不通）
        cmd += ["--model-path", str(ROOT / c["model_path"])]
    return cmd + list(c.get("extra_args", []))


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


# ---------------------------------------------------------------- 进程管理

class Managed:
    """一个被托管的子进程：启动、健康检查、日志转发、强制结束。"""

    def __init__(self, name: str, cmd: list[str], health_url: str, hub: Hub,
                 external: bool = False, env: dict[str, str] | None = None):
        self.name = name
        self.cmd = cmd
        self.health_url = health_url
        self.hub = hub
        self.external = external      # 由 LM Studio 之类的外部程序托管，我们只做健康检查
        self.env = env or {}
        self.healthy = False
        self.proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        if self.external:
            return self.healthy
        return self.proc is not None and self.proc.poll() is None

    async def start(self) -> None:
        if self.external or self.running:
            return
        await self.hub.log(f"[{self.name}] 启动：{' '.join(self.cmd)}")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        env = os.environ.copy()
        env.update(self.env)
        try:
            self.proc = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, cwd=str(ROOT), creationflags=flags, env=env,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"找不到可执行文件 {self.cmd[0]}。"
                f"请在 config.json 里把 {self.name}.binary 改成完整路径。"
            )
        threading.Thread(
            target=self._pump, args=(asyncio.get_running_loop(),), daemon=True
        ).start()

    def _pump(self, loop: asyncio.AbstractEventLoop) -> None:
        """子进程日志转发到网页。llama-server 的 eval time 就是从这里看的。"""
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            text = raw.decode("utf-8", "replace").rstrip()
            if not text:
                continue
            try:
                asyncio.run_coroutine_threadsafe(
                    self.hub.log(f"[{self.name}] {text}"), loop
                )
            except RuntimeError:
                return

    def stop(self) -> None:
        self.healthy = False
        if self.external or self.proc is None or self.proc.poll() is not None:
            return          # 外部托管的服务不归我们关
        if os.name == "nt":
            # terminate 杀不掉子进程树，llama-server 会残留占着显存
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True)
        else:
            self.proc.terminate()
        self.proc = None

    async def wait_healthy(self, timeout: float | None = None) -> bool:
        """等就绪。自己启动的要加载模型可能几分钟；外部服务没开就该立刻报错。"""
        if timeout is None:
            timeout = 20.0 if self.external else 600.0
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.proc is not None and self.proc.poll() is not None:
                    await self.hub.log(f"[{self.name}] 进程已退出，启动失败", "error")
                    return False
                try:
                    resp = await client.get(self.health_url)
                    if resp.status_code < 500:
                        self.healthy = True
                        await self.hub.log(f"[{self.name}] 就绪")
                        return True
                except Exception:
                    pass
                await asyncio.sleep(1.5)
        if self.external:
            await self.hub.log(
                f"[{self.name}] 连不上 {self.health_url}，"
                "请确认 LM Studio 已启动、模型已加载、本地服务器开关已打开", "error")
        else:
            await self.hub.log(f"[{self.name}] 等待就绪超时", "error")
        return False


# ---------------------------------------------------------------- 转写整形

SENTENCE_END = "。！？…!?"


def parse_ts(value) -> float | None:
    """WhisperLiveKit 的时间戳是 '0:00:23.29' 这种字符串，不是秒数。"""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        seconds = 0.0
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
        return seconds
    except ValueError:
        return None


def to_sentences(lines: list[dict]) -> list[dict]:
    """把 ASR 的长行切成逐句，并插值出每句的时间戳。

    WhisperLiveKit 只在换说话人或长静音时才另起一行，同一个人连讲两分钟就是
    一整行，展示成一大块没法用，也没法「点某句跳到录音那一处」。这里按句末
    标点切开，再按字符占比在该行的 [start, end] 区间里插值。插值不如逐词
    对齐精确，但用来定位听回放足够，而 ASR 本身并不给逐词时间戳。
    """
    out: list[dict] = []
    for line in lines:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        beg, fin = parse_ts(line.get("start")), parse_ts(line.get("end"))
        speaker = line.get("speaker")

        chunks, buf = [], ""
        for ch in text:
            buf += ch
            if ch in SENTENCE_END:
                if buf.strip():
                    chunks.append(buf.strip())
                buf = ""
        if buf.strip():
            chunks.append(buf.strip())

        total = sum(len(c) for c in chunks) or 1
        span = (fin - beg) if (beg is not None and fin is not None and fin > beg) else None
        pos = 0
        for c in chunks:
            if beg is None:
                t0 = t1 = None
            elif span is None:
                t0 = t1 = beg
            else:
                t0 = beg + span * pos / total
                t1 = beg + span * (pos + len(c)) / total
            out.append({"speaker": speaker, "text": c, "beg": t0, "end": t1})
            pos += len(c)
    return out


# ---------------------------------------------------------------- 主流程

class App:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.hub = Hub()
        self.minutes = mn.Minutes()
        self.state = "idle"          # idle | starting | running | stopping
        self.services: dict[str, Managed] = {}
        self.task: asyncio.Task | None = None
        self.worker: asyncio.Task | None = None
        self.llm: mn.LocalLLM | None = None
        self.transcript_lines: list[dict] = []
        self.buffer_text = ""
        self.meeting: mt.Meeting | None = None
        self.recorder: mt.WavRecorder | None = None
        self._last_persist = 0.0

    # ---- 状态推送

    def status(self) -> dict:
        meeting = None
        if self.meeting is not None:
            meta = self.meeting.meta()
            meeting = {
                "id": self.meeting.id,
                "title": meta.get("title", ""),
                "started_at": meta.get("started_at"),
                "audio_seconds": self.recorder.seconds if self.recorder else 0.0,
            }
        return {
            "type": "status",
            "state": self.state,
            "services": {
                name: ("running" if svc.running else "stopped")
                for name, svc in self.services.items()
            },
            "meeting": meeting,
        }

    async def push_status(self) -> None:
        await self.hub.broadcast(self.status())

    async def push_minutes(self) -> None:
        await self.hub.broadcast({"type": "minutes", **self.minutes.to_json()})

    # ---- 启停

    async def start(self) -> None:
        if self.state != "idle":
            return
        self.state = "starting"
        # 每场会议从空白开始。之前不清空，上一场的条目会混进这一场的纪要里。
        self.minutes = mn.Minutes()
        self.transcript_lines = []
        self.buffer_text = ""
        await self.push_minutes()
        await self.push_status()
        try:
            # 国内直连 HuggingFace 会卡在下载模型这步，给子进程注入镜像地址
            env: dict[str, str] = {}
            mirror = self.cfg.get("network", {}).get("hf_endpoint")
            if mirror:
                env["HF_ENDPOINT"] = mirror
                await self.hub.log(f"使用 HuggingFace 镜像 {mirror}")

            for name, builder in (("llm", llm_cmd), ("asr", asr_cmd)):
                conf = self.cfg[name]
                external = not conf.get("manage", True)
                health = (f"http://127.0.0.1:{conf['port']}"
                          f"{conf.get('health_path', '/')}")
                svc = Managed(name, builder(self.cfg), health, self.hub, external, env)
                self.services[name] = svc
                if external:
                    await self.hub.log(f"[{name}] 外部托管，检查 {health} …")
                else:
                    await svc.start()
                await self.push_status()
                if not await svc.wait_healthy():
                    raise RuntimeError(f"{name} 未能就绪")
                await self.push_status()

            self.meeting = mt.create()
            self.recorder = mt.WavRecorder(self.meeting.audio_path)
            self._last_persist = 0.0
            await self.hub.log(f"本场会议归档到 meetings/{self.meeting.id}/")

            self.task = asyncio.create_task(self.pipeline())
            self.state = "running"
            await self.push_status()
        except Exception as exc:
            await self.hub.log(f"启动失败：{exc}", "error")
            await self.stop()

    async def stop(self) -> None:
        self.state = "stopping"
        await self.push_status()
        for task in (self.task, self.worker):
            if task and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.task = self.worker = None
        if self.llm:
            await self.llm.aclose()
            self.llm = None
        for svc in self.services.values():
            svc.stop()
        self.services.clear()
        await self.finalize_meeting()
        self.state = "idle"
        await self.push_status()
        await self.hub.log("已停止")

    # ---- 归档

    def persist(self, meeting: mt.Meeting | None = None) -> None:
        """把转写和纪要落到本场会议目录。中途也写，强杀不至于全丢。"""
        meeting = meeting or self.meeting
        if meeting is None:
            return
        meta = meeting.meta()
        started = meta.get("started_at")
        subtitle = (f"_{datetime.fromtimestamp(started):%Y-%m-%d %H:%M} 开始_"
                    if started else None)
        try:
            meeting.write_transcript(self.transcript_lines)
            meeting.write_minutes(
                mn.render_markdown(self.minutes, meta.get("title", "会议纪要"), subtitle),
                self.minutes.to_json(),
            )
        except OSError:
            pass

    async def finalize_meeting(self) -> None:
        if self.meeting is None:
            return
        meeting, recorder = self.meeting, self.recorder
        self.meeting, self.recorder = None, None
        if recorder is not None:
            recorder.close()
        self.transcript_lines = [
            l for l in self.transcript_lines if (l.get("text") or "").strip()
        ]
        # 启动失败或点开始又立刻停止，会留下一个空壳目录，别让它堆进历史列表
        if not self.transcript_lines and (recorder is None or recorder.seconds < 1.0):
            mt.remove(meeting.id)
            await self.hub.log(f"本场没有任何内容，已丢弃 meetings/{meeting.id}/")
            return
        self.persist(meeting)
        started = meeting.meta().get("started_at") or time.time()
        ended = time.time()
        audio_seconds = round(recorder.seconds, 1) if recorder else 0.0
        meeting.write_meta(
            ended_at=ended,
            duration=round(ended - started, 1),
            line_count=len(self.transcript_lines),
            item_count=len(self.minutes.items),
            audio_seconds=audio_seconds,
        )
        await self.hub.log(
            f"已归档 meetings/{meeting.id}/：录音 {mt.fmt_ts(audio_seconds)}"
            f"，转写 {len(self.transcript_lines)} 句，纪要 {len(self.minutes.items)} 条"
        )

    # ---- 音频 -> 转写 -> 摘要

    async def resolve_model(self, base_url: str, configured: str) -> str:
        """问服务端实际加载了哪个模型。

        LM Studio 给本地 GGUF 分配的 id 要它索引后才确定，写死在配置里很容易对不上，
        所以对不上时直接用服务端报告的第一个，省掉手工抄写这一步。
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(base_url.rstrip("/") + "/models")
                ids = [m["id"] for m in resp.json().get("data", []) if m.get("id")]
        except Exception as exc:
            await self.hub.log(f"查询模型列表失败，沿用配置值：{exc}")
            return configured
        if not ids:
            return configured
        if configured in ids:
            return configured
        await self.hub.log(f"配置的 {configured} 未加载，改用服务端的 {ids[0]}")
        return ids[0]

    async def pipeline(self) -> None:
        s = self.cfg["summary"]
        base = f"http://127.0.0.1:{self.cfg['llm']['port']}/v1"
        model = await self.resolve_model(base, self.cfg["llm"]["model_name"])
        self.llm = mn.LocalLLM(base, model)
        queue: asyncio.Queue = asyncio.Queue()
        self.worker = asyncio.create_task(mn.summarizer(
            queue, self.llm, self.minutes, self.cfg["out"],
            s["overview_every"], on_log=self.hub.log, on_update=self.push_minutes,
        ))

        ws_url = (f"ws://127.0.0.1:{self.cfg['asr']['port']}/asr"
                  f"?language={self.cfg['asr']['language']}")
        dev = self.cfg["audio"]["device"]
        loopback = self.cfg["audio"]["loopback"]
        auto_gain = self.cfg["audio"].get("auto_gain", True)
        rec = self.recorder

        async with websockets.connect(ws_url, max_size=None) as ws:
            await self.hub.log(f"已连接转写服务 {ws_url}")
            sender = asyncio.create_task(
                audio.send_audio(ws, dev, loopback, on_log=self.hub.log,
                                 auto_gain=auto_gain,
                                 tap=(rec.write if rec is not None else None))
            )
            try:
                await self.receive(ws, queue, s)
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    async def receive(self, ws, queue: asyncio.Queue, s: dict) -> None:
        """接 full 模式快照。只把已确认（非最后一行）的文本送去摘要。"""
        consumed, pending = 0, ""
        last_flush = time.monotonic()

        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            lines = [l for l in msg.get("lines", []) if isinstance(l, dict)]
            self.transcript_lines = to_sentences(lines)
            self.buffer_text = str(msg.get("buffer_transcription", "") or "")
            await self.hub.broadcast({
                "type": "transcript",
                "lines": self.transcript_lines,
                "buffer": self.buffer_text,
            })

            # 按句而不是按行取「已确认」的部分：ASR 只在换人或长静音时才换行，
            # 按行取的话一个人连讲十分钟都攒不出一次摘要，实时性就没了。
            # 最后一句还可能被修正，留着不消费。
            texts = [l["text"] for l in self.transcript_lines]
            stable = "".join(texts[:-1]) if len(texts) > 1 else ""
            if len(stable) > consumed:
                pending += stable[consumed:]
                consumed = len(stable)

            elapsed = time.monotonic() - last_flush
            if pending and (len(pending) >= s["min_chars"] or elapsed >= s["max_wait"]):
                await self.hub.log(f"送入摘要 {len(pending)} 字")
                await mn.put_chunk(queue, pending, s["max_pending"], self.hub.log)
                pending, last_flush = "", time.monotonic()

            # 转写快照会被不断修订，所以整份重写而不是追加；节流到 5 秒一次。
            now = time.monotonic()
            if now - self._last_persist >= 5.0:
                self._last_persist = now
                self.persist()

        if pending:
            await queue.put(pending)


APP = App()
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
        await APP.hub.log(f"枚举音频设备失败：{exc}", "error")
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
    await APP.hub.log("配置已保存" + ("（重启后生效）" if "n_cpu_moe" in data else ""))
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
    APP.hub.clients.add(ws)
    await ws.send_json(APP.status())
    await ws.send_json({"type": "minutes", **APP.minutes.to_json()})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        APP.hub.clients.discard(ws)


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
