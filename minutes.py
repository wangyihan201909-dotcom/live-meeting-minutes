"""纪要状态机与摘要逻辑。

核心：不做全量重摘，而是「抽取 + 有状态合并」两步。
抽取只看新增片段，合并只看当前条目列表（id + 一句话）。
送进模型的上下文长度恒定，条目 id 稳定，纪要不会整篇重写乱跳。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx

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

    def to_json(self) -> dict:
        """给前端渲染用。"""
        return {
            "overview": self.overview,
            "groups": [
                {
                    "type": t,
                    "label": label,
                    "items": [
                        {"id": i.id, "text": i.text, "owner": i.owner, "hits": i.hits}
                        for i in self.by_type(t)
                    ],
                }
                for t, label in TYPES.items()
            ],
            "total": len(self.items),
        }


# ---------------------------------------------------------------- 本地模型

class LocalLLM:
    """OpenAI 兼容端点客户端，用 JSON Schema 约束解码，并关掉思考模式。

    两个实测踩过的坑：

    1) `reasoning_effort: "none"` 必须带。Qwen3.6 这类推理模型默认会先思考几百个
       token 才作答，全部进 reasoning_content，content 是空的。在 12 tok/s 的机器上
       光思考就要 48 秒且撞 token 上限，等于永远拿不到结果。实测 LM Studio 不认
       `/no_think` 提示词，也不认 `chat_template_kwargs.enable_thinking=false`，
       只有 `reasoning_effort` 生效：48 秒失败 -> 5.7 秒成功。

    2) response_format 各家不一致。LM Studio 只认 json_schema / text，传
       json_object 会 400；llama.cpp 和 Ollama 认 json_object。

    所以这里带全部特性发出，遇到 400 就按阶梯逐项摘掉，换任何服务端都能跑。
    """

    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.mode = "json_schema"        # -> json_object -> text
        self.no_reasoning = True
        self.client = httpx.AsyncClient(timeout=timeout)

    def _payload(self, system: str, user: str, schema: dict | None) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.no_reasoning:
            payload["reasoning_effort"] = "none"
        if schema and self.mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema},
            }
        elif schema and self.mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _degrade(self, detail: str) -> str | None:
        """400 时按阶梯摘掉一项特性，返回变更说明；无可摘则 None。"""
        if "reasoning" in detail.lower() and self.no_reasoning:
            self.no_reasoning = False
            return "去掉 reasoning_effort"
        if self.mode == "json_schema":
            self.mode = "json_object"
            return "response_format 降级到 json_object"
        if self.mode == "json_object":
            self.mode = "text"
            return "response_format 降级到纯文本"
        if self.no_reasoning:
            self.no_reasoning = False
            return "去掉 reasoning_effort"
        return None

    async def json_call(self, system: str, user: str, schema: dict | None = None,
                        on_log=None, retries: int = 1) -> dict | None:
        attempt = 0
        while attempt <= retries:
            try:
                resp = await self.client.post(self.url, json=self._payload(system, user, schema))
                if resp.status_code == 400:
                    change = self._degrade(resp.text)
                    if change:
                        if on_log:
                            await on_log(f"服务端拒绝请求，{change} 后重试")
                        continue          # 降级不计入重试次数
                    if on_log:
                        await on_log(f"服务端 400 且无可降级项：{resp.text[:120]}")
                    return None
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                raw = msg.get("content") or ""
                if not raw and msg.get("reasoning_content"):
                    if on_log:
                        await on_log("模型只输出了推理内容，正文为空——思考模式没关掉")
            except Exception as exc:          # 模型服务抖动不能拖垮转写
                if on_log:
                    await on_log(f"模型调用失败({attempt})：{exc}")
                attempt += 1
                await asyncio.sleep(1.0)
                continue
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed
            if on_log:
                await on_log(f"模型输出不是合法 JSON，已丢弃：{raw[:100]}")
            attempt += 1
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


# ---------------------------------------------------------------- 提示词

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
new —— 刚从最新转写里抽到的要点，每条自带 index

对 new 里的每一条各输出一个操作，index 必须原样抄用那一条自带的 index，不要自己编：
- {"op":"add","index":0}                     全新的内容，新增一条
- {"op":"merge","index":1,"id":"t3"}         和已有条目说的是同一件事，忽略即可
- {"op":"update","index":2,"id":"a1","text":"新的表述"}  是对已有条目的补充或修正，用新表述替换

id 必须是 current 里真实存在的 id。
判断标准是“说的是不是同一件事”，措辞不同不算不同。宁可 merge，不要制造重复条目。
new 有几条就输出几个操作，不多不少。
只输出 JSON，不要任何解释：{"ops":[...]}"""

OVERVIEW_SYSTEM = """根据下面的会议纪要条目，写一段不超过 80 字的中文概述，说明这场会议到目前为止在谈什么、进展到哪。
只输出 JSON：{"overview":"..."}"""


# ---------------------------------------------------------------- 输出结构约束
# 支持约束解码的服务端会照这个 schema 生成，从根上杜绝非法 JSON

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(TYPES)},
                    "text": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                },
                "required": ["type", "text"],
            },
        }
    },
    "required": ["points"],
}

MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "merge", "update"]},
                    "index": {"type": "integer"},
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["op", "index"],
            },
        }
    },
    "required": ["ops"],
}

OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {"overview": {"type": "string"}},
    "required": ["overview"],
}


# ---------------------------------------------------------------- 摘要循环

async def summarizer(
    queue: asyncio.Queue,
    llm: LocalLLM,
    minutes: Minutes,
    out_path: str,
    overview_every: int,
    on_log=None,
    on_update=None,
) -> None:
    rounds = 0
    while True:
        chunk = await queue.get()
        if chunk is None:
            break

        extracted = await llm.json_call(
            EXTRACT_SYSTEM, f"转写片段：\n{chunk}", EXTRACT_SCHEMA, on_log)
        points = (extracted or {}).get("points") or []
        points = [p for p in points if isinstance(p, dict) and p.get("text")]
        if not points:
            queue.task_done()
            continue

        current = minutes.digest()
        if current:
            # 每条带上显式 index 让模型照抄。让它自己数数组下标，小模型会报出
            # 越界的 index，去重就失效了（保护逻辑会挡住脏数据，但重复条目照样产生）
            indexed = [{"index": i, **p} for i, p in enumerate(points)]
            merged = await llm.json_call(
                MERGE_SYSTEM,
                json.dumps({"current": current, "new": indexed}, ensure_ascii=False),
                MERGE_SCHEMA, on_log,
            )
            ops = (merged or {}).get("ops") or []
        else:
            ops = [{"op": "add", "index": i} for i in range(len(points))]

        apply_ops(minutes, points, ops)

        rounds += 1
        if overview_every and rounds % overview_every == 0 and minutes.items:
            ov = await llm.json_call(
                OVERVIEW_SYSTEM, json.dumps(minutes.digest(), ensure_ascii=False),
                OVERVIEW_SCHEMA, on_log,
            )
            if ov and ov.get("overview"):
                minutes.overview = str(ov["overview"]).strip()

        write_markdown(minutes, out_path)
        if on_update:
            await on_update()
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


async def put_chunk(queue: asyncio.Queue, text: str, max_pending: int, on_log=None) -> None:
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
            if item is None:                  # 结束哨兵不能吞掉
                await queue.put(None)
                break
            merged.insert(0, item)
        text = "\n".join(merged)
        if on_log:
            await on_log(f"摘要落后，合并积压片段 → {len(text)} 字")
    await queue.put(text)


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
