#!/usr/bin/env python3
"""
实时滚动会议纪要 —— 全本地，无云端 API。

数据流：
    声卡 (WASAPI loopback / 麦克风)
      --> 本脚本采集 16k s16le PCM --> WhisperLiveKit + Qwen3-ASR (ws://.../asr)
      <-- 持续回传已确认的转写行
      --> 增量抽取 + 有状态合并 --> 本地 MoE 模型 (OpenAI 兼容端点)
      --> live_minutes.md 持续覆盖刷新

两个关键设计：

1) 摘要不是每次把全量转写重新喂给模型，而是「抽取 + 有状态合并」两步：
   抽取只看新增的那一段，合并只看当前条目列表（id + 一句话）。
   送进模型的上下文长度恒定，会开三小时也不涨；条目 id 稳定，纪要不会整篇重写乱跳。

2) 音频由本脚本采集并推送。WhisperLiveKit 每个 WS 连接是独立会话、不广播转写，
   所以不能另开浏览器采音频、脚本这边干等——必须自己当那个唯一的客户端。

依赖：
    pip install websockets httpx sounddevice
服务端需以 PCM 直通模式启动（跳过 ffmpeg）：
    wlk --backend qwen3-streaming --language zh --pcm-input
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000          # WhisperLiveKit --pcm-input 约定：16k / 单声道 / s16le
BLOCK_SECONDS = 0.5

# ---------------------------------------------------------------- 纪要状态

TYPES = {
    "topic": "议题",
    "decision": "决议",
    "action": "待办",
    "risk": "风险",
    "question": "待确认问题",
}


@dataclass
class Item:
    id: str
    type: str
    text: str
    owner: str | None = None
    hits: int = 1
    first_seen: float = field(default_factory=time.time)


@dataclass
class Minutes:
    items: dict[str, Item] = field(default_factory=dict)
    overview: str = ""
    _seq: int = 0

    def new_id(self, type_: str) -> str:
        self._seq += 1
        return f"{type_[:1]}{self._seq}"

    def by_type(self, type_: str) -> list[Item]:
        return sorted(
            (i for i in self.items.values() if i.type == type_),
            key=lambda i: i.first_seen,
        )

    def digest(self, per_type: int = 12) -> list[dict]:
        """喂给合并步骤的精简视图：只给 id / type / text。"""
        out: list[dict] = []
        for t in TYPES:
            for item in self.by_type(t)[-per_type:]:
                out.append({"id": item.id, "type": item.type, "text": item.text})
        return out


# ---------------------------------------------------------------- 本地模型

class LocalLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    async def json_call(self, system: str, user: str, retries: int = 1) -> dict | None:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        for attempt in range(retries + 1):
            try:
                resp = await self.client.post(self.url, json=payload)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:              # 模型服务抖动不能拖垮转写
                log(f"[llm] 调用失败({attempt}): {exc}")
                await asyncio.sleep(1.0)
                continue
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            log(f"[llm] 输出不是合法 JSON，丢弃：{raw[:120]!r}")
        return None

    async def aclose(self) -> None:
        await self.client.aclose()


def extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- 两步提示词

EXTRACT_SYSTEM = """你是中文会议记录助手。用户给你一段会议转写片段（可能不完整、有口语和识别错误）。
只从这段文字里抽取确实说过的要点，禁止推测、补全或润色成没说过的内容。

按下面的类型分类：
- topic：正在讨论的议题
- decision：已经拍板的结论
- action：要有人去做的事，尽量填 owner
- risk：风险、阻塞、担忧
- question：悬而未决、需要后续确认的问题

只输出 JSON，不要任何解释：
{"points":[{"type":"topic","text":"一句话，20字以内","owner":null}]}
片段里没有值得记录的内容（寒暄、口水话、静默）就输出 {"points":[]}。"""

MERGE_SYSTEM = """你在维护一份持续更新的会议纪要。用户会给你：
current —— 纪要里已有的条目（含 id）
new —— 刚从最新转写里抽到的要点

对 new 里的每一条，判断它和 current 的关系，输出一个操作：
- {"op":"add","index":0}                     全新的内容，新增一条
- {"op":"merge","index":1,"id":"t3"}         和已有条目说的是同一件事，忽略即可
- {"op":"update","index":2,"id":"a1","text":"新的表述"}  是对已有条目的补充或修正，用新表述替换

判断标准是“说的是不是同一件事”，措辞不同不算不同。宁可 merge，不要制造重复条目。
只输出 JSON，不要任何解释：{"ops":[...]}"""

OVERVIEW_SYSTEM = """根据下面的会议纪要条目，写一段不超过 80 字的中文概述，说明这场会议到目前为止在谈什么、进展到哪。
只输出 JSON：{"overview":"..."}"""


# ---------------------------------------------------------------- 摘要循环

async def summarizer(
    queue: asyncio.Queue[str | None],
    llm: LocalLLM,
    minutes: Minutes,
    out_path: str,
    overview_every: int,
) -> None:
    rounds = 0
    while True:
        chunk = await queue.get()
        if chunk is None:
            break

        extracted = await llm.json_call(EXTRACT_SYSTEM, f"转写片段：\n{chunk}")
        points = (extracted or {}).get("points") or []
        points = [p for p in points if isinstance(p, dict) and p.get("text")]
        if not points:
            queue.task_done()
            continue

        current = minutes.digest()
        if current:
            merged = await llm.json_call(
                MERGE_SYSTEM,
                json.dumps({"current": current, "new": points}, ensure_ascii=False),
            )
            ops = (merged or {}).get("ops") or []
        else:
            ops = [{"op": "add", "index": i} for i in range(len(points))]

        apply_ops(minutes, points, ops)

        rounds += 1
        if overview_every and rounds % overview_every == 0 and minutes.items:
            ov = await llm.json_call(
                OVERVIEW_SYSTEM, json.dumps(minutes.digest(), ensure_ascii=False)
            )
            if ov and ov.get("overview"):
                minutes.overview = str(ov["overview"]).strip()

        write_markdown(minutes, out_path)
        log(f"[纪要] 已更新 {out_path}（{len(minutes.items)} 条）")
        queue.task_done()


def apply_ops(minutes: Minutes, points: list[dict], ops: list[dict]) -> None:
    handled: set[int] = set()
    for op in ops:
        if not isinstance(op, dict):
            continue
        idx = op.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(points)):
            continue
        handled.add(idx)
        kind = op.get("op")
        target = minutes.items.get(str(op.get("id", "")))
        if kind == "merge" and target:
            target.hits += 1
        elif kind == "update" and target:
            target.hits += 1
            if op.get("text"):
                target.text = str(op["text"]).strip()
        else:
            add_point(minutes, points[idx])
    # 模型漏判的一律按新增处理，宁可多记也不要丢
    for i, point in enumerate(points):
        if i not in handled:
            add_point(minutes, point)


def add_point(minutes: Minutes, point: dict) -> None:
    type_ = point.get("type") if point.get("type") in TYPES else "topic"
    item_id = minutes.new_id(type_)
    minutes.items[item_id] = Item(
        id=item_id,
        type=type_,
        text=str(point["text"]).strip(),
        owner=(str(point["owner"]).strip() if point.get("owner") else None),
    )


# ---------------------------------------------------------------- 输出

def write_markdown(minutes: Minutes, path: str) -> None:
    lines = [
        "# 实时会议纪要",
        "",
        f"_更新于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]
    if minutes.overview:
        lines += ["## 概述", "", minutes.overview, ""]
    for type_, label in TYPES.items():
        items = minutes.by_type(type_)
        if not items:
            continue
        lines += [f"## {label}", ""]
        for item in items:
            owner = f"**@{item.owner}** " if item.owner else ""
            mark = " ⭐" if item.hits >= 3 else ""
            lines.append(f"- {owner}{item.text}{mark}")
        lines.append("")

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    os.replace(tmp, path)        # 原子替换，实时预览不会读到半截文件


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- 音频采集

def list_devices() -> None:
    """打印设备表。找会议声音要认准 WASAPI 的输出设备 + --loopback。"""
    print(sd.query_devices())
    print("\n提示：只采麦克风只能录到你自己说话。要录别人的发言，")
    print("请选一个 WASAPI 输出设备（扬声器/耳机）的编号，并加上 --loopback。")


def open_stream(device: int | None, loopback: bool, on_block) -> sd.RawInputStream:
    extra = None
    if loopback:
        try:
            extra = sd.WasapiSettings(loopback=True)
        except (AttributeError, TypeError) as exc:
            raise SystemExit(
                f"当前 sounddevice/PortAudio 不支持 WASAPI loopback：{exc}\n"
                "升级 sounddevice，或改用 VB-Audio Virtual Cable 之类的虚拟声卡。"
            )
    return sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
        device=device,
        channels=1,
        dtype="int16",
        extra_settings=extra,
        callback=on_block,
    )


async def send_audio(ws, device: int | None, loopback: bool) -> None:
    """采集 PCM 并持续推给服务端。回调在 PortAudio 线程，转交事件循环。"""
    loop = asyncio.get_running_loop()
    blocks: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)

    def on_block(indata, _frames, _t, status) -> None:
        if status:
            log(f"[音频] {status}")
        try:
            loop.call_soon_threadsafe(blocks.put_nowait, bytes(indata))
        except RuntimeError:
            pass                                  # 事件循环已关闭，退出中

    with open_stream(device, loopback, on_block):
        log(f"[音频] 开始采集 device={device} loopback={loopback}")
        while True:
            await ws.send(await blocks.get())


# ---------------------------------------------------------------- 转写接收

async def put_chunk(queue: asyncio.Queue[str | None], text: str, max_pending: int) -> None:
    """背压：模型跟不上说话速度时合并积压片段，而不是让队列无限增长。

    会议不会等模型。队列一旦开始堆积就再也追不上，
    最后写出来的是二十分钟前的内容——宁可粒度粗一点也要跟住当下。
    """
    if queue.qsize() >= max_pending:
        merged = [text]
        while not queue.empty():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            queue.task_done()
            if item is None:                      # 结束哨兵不能吞掉
                await queue.put(None)
                break
            merged.insert(0, item)
        text = "\n".join(merged)
        log(f"[背压] 摘要落后，合并积压片段 -> {len(text)} 字")
    await queue.put(text)


async def receive_transcript(
    ws,
    queue: asyncio.Queue[str | None],
    min_chars: int,
    max_wait: float,
    transcript_path: str | None,
    max_pending: int,
) -> None:
    """接 WhisperLiveKit 的 full 模式快照，只消费已确认（非最后一行）的文本。"""
    consumed = 0
    pending = ""
    last_flush = time.monotonic()

    async for raw in ws:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        texts = [str(l.get("text", "")) for l in msg.get("lines", []) if isinstance(l, dict)]
        # 最后一行还可能被修正，只消费它之前的部分
        stable = "".join(texts[:-1]) if len(texts) > 1 else ""
        if len(stable) > consumed:
            delta = stable[consumed:]
            consumed = len(stable)
            pending += delta
            if transcript_path:
                with open(transcript_path, "a", encoding="utf-8") as fh:
                    fh.write(delta)

        elapsed = time.monotonic() - last_flush
        if pending and (len(pending) >= min_chars or elapsed >= max_wait):
            log(f"[asr] 送入摘要 {len(pending)} 字")
            await put_chunk(queue, pending, max_pending)
            pending, last_flush = "", time.monotonic()

    if pending:
        await queue.put(pending)


async def run_session(args, queue: asyncio.Queue[str | None]) -> None:
    async with websockets.connect(args.ws, max_size=None) as ws:
        log(f"[asr] 已连接 {args.ws}")
        sender = asyncio.create_task(send_audio(ws, args.device, args.loopback))
        try:
            await receive_transcript(
                ws, queue, args.min_chars, args.max_wait,
                args.transcript, args.max_pending,
            )
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)


# ---------------------------------------------------------------- 入口

async def main() -> int:
    ap = argparse.ArgumentParser(description="全本地实时滚动会议纪要")
    ap.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    ap.add_argument("--device", type=int, default=None, help="音频设备编号")
    ap.add_argument("--loopback", action="store_true",
                    help="采系统输出而不是麦克风（Windows WASAPI），录别人的发言必须开")
    ap.add_argument("--ws", default="ws://localhost:8000/asr?language=zh",
                    help="WhisperLiveKit WebSocket 地址，服务端需带 --pcm-input")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8080/v1",
                    help="OpenAI 兼容端点。llama-server 默认 8080，Ollama 是 11434")
    ap.add_argument("--llm-model", default="qwen3.6-35b-a3b",
                    help="本地模型名，需与推理服务里注册的名字一致")
    ap.add_argument("--out", default="live_minutes.md", help="纪要输出文件")
    ap.add_argument("--transcript", default=None, help="可选：同时落地全量转写")
    ap.add_argument("--min-chars", type=int, default=180, help="攒够多少字触发一次抽取")
    ap.add_argument("--max-wait", type=float, default=45.0,
                    help="最多等多少秒必定触发一次，冷场也能出结果")
    ap.add_argument("--overview-every", type=int, default=4,
                    help="每 N 轮重写一次概述，设 0 关闭")
    ap.add_argument("--max-pending", type=int, default=2,
                    help="队列积压超过这个数就合并待处理片段")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    minutes = Minutes()
    llm = LocalLLM(args.llm_url, args.llm_model)
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    write_markdown(minutes, args.out)
    worker = asyncio.create_task(
        summarizer(queue, llm, minutes, args.out, args.overview_every)
    )
    try:
        await run_session(args, queue)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log(f"[asr] 会话结束：{exc}")
    finally:
        await queue.put(None)
        await worker
        await llm.aclose()
        write_markdown(minutes, args.out)
        log(f"[纪要] 已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
