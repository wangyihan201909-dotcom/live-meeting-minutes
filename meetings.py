"""会议归档：一场会议一个目录，存录音、带时间戳的转写、纪要。

    meetings/<id>/
        meta.json        标题、起止时间、时长、统计
        audio.wav        16kHz 单声道，边开会边写
        transcript.json  带时间戳的逐句转写
        minutes.md       渲染好的纪要，会后直接拿这个文件
        minutes.json     结构化纪要，导出和重新渲染用

只用标准库。归档是会后唯一的凭据，所以这里的写入都走「先写临时文件再原子替换」，
断电或强杀不会留下半截 JSON。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEETINGS_DIR = ROOT / "meetings"

SAMPLE_RATE = 16000          # 与 audio.TARGET_RATE 一致
SAMPLE_WIDTH = 2             # s16le
CHANNELS = 1


# ---------------------------------------------------------------- 原子写

def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


# ---------------------------------------------------------------- 录音

class WavRecorder:
    """边采边写 WAV。

    `wave` 模块只在 close() 时回填头部的长度字段，中途被强杀的话头里写的是 0，
    多数播放器会认成空文件。所以另外记一份 sidecar 字节数，
    并提供 `repair` 在读取时按实际文件大小把头补回来——录音丢了就没法复盘了。
    """

    def __init__(self, path: Path):
        self.path = path
        self._wav: wave.Wave_write | None = None
        self.bytes_written = 0
        try:
            self._wav = wave.open(str(path), "wb")
            self._wav.setnchannels(CHANNELS)
            self._wav.setsampwidth(SAMPLE_WIDTH)
            self._wav.setframerate(SAMPLE_RATE)
        except OSError:
            self._wav = None

    def write(self, pcm: bytes) -> None:
        if self._wav is None or not pcm:
            return
        try:
            self._wav.writeframes(pcm)
            self.bytes_written += len(pcm)
        except (OSError, ValueError):
            self.close()

    @property
    def seconds(self) -> float:
        return self.bytes_written / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)

    def close(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            except (OSError, ValueError):
                pass
            self._wav = None


def repair_wav(path: Path) -> bool:
    """头里长度为 0 但文件有数据时，按实际大小回填 RIFF/data 长度。返回是否修过。"""
    try:
        size = path.stat().st_size
        if size <= 44:
            return False
        with open(path, "r+b") as fh:
            fh.seek(0)
            if fh.read(4) != b"RIFF":
                return False
            fh.seek(40)
            (data_len,) = struct.unpack("<I", fh.read(4))
            if data_len != 0:
                return False
            fh.seek(4)
            fh.write(struct.pack("<I", size - 8))
            fh.seek(40)
            fh.write(struct.pack("<I", size - 44))
        return True
    except (OSError, struct.error):
        return False


# ---------------------------------------------------------------- 归档

def _new_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_id(meeting_id: str) -> str | None:
    """只接受本模块生成的形状，挡住 ../ 之类的路径穿越。"""
    return meeting_id if re.fullmatch(r"\d{8}-\d{6}", meeting_id or "") else None


@dataclass
class Meeting:
    id: str
    dir: Path

    @property
    def audio_path(self) -> Path:
        return self.dir / "audio.wav"

    def meta(self) -> dict:
        return _read_json(self.dir / "meta.json", {})

    def write_meta(self, **patch) -> dict:
        meta = self.meta()
        meta.update(patch)
        _atomic_write(self.dir / "meta.json",
                      json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    def write_transcript(self, lines: list[dict]) -> None:
        _atomic_write(self.dir / "transcript.json",
                      json.dumps(lines, ensure_ascii=False, indent=1))

    def transcript(self) -> list[dict]:
        return _read_json(self.dir / "transcript.json", [])

    def write_minutes(self, markdown: str, structured: dict) -> None:
        _atomic_write(self.dir / "minutes.md", markdown)
        _atomic_write(self.dir / "minutes.json",
                      json.dumps(structured, ensure_ascii=False, indent=2))

    def minutes_md(self) -> str:
        try:
            return (self.dir / "minutes.md").read_text(encoding="utf-8")
        except OSError:
            return ""

    def minutes_json(self) -> dict:
        return _read_json(self.dir / "minutes.json", {})


def create(title: str | None = None) -> Meeting:
    MEETINGS_DIR.mkdir(exist_ok=True)
    mid = _new_id()
    # 同一秒内重复开始会撞 id，退到带后缀的秒
    while (MEETINGS_DIR / mid).exists():
        time.sleep(1.0)
        mid = _new_id()
    d = MEETINGS_DIR / mid
    d.mkdir()
    m = Meeting(mid, d)
    started = time.time()
    m.write_meta(
        id=mid,
        title=title or datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M 会议"),
        started_at=started,
        ended_at=None,
        duration=0.0,
        line_count=0,
        item_count=0,
        audio_seconds=0.0,
    )
    m.write_transcript([])
    return m


def get(meeting_id: str) -> Meeting | None:
    mid = _safe_id(meeting_id)
    if not mid:
        return None
    d = MEETINGS_DIR / mid
    return Meeting(mid, d) if d.is_dir() else None


def listing(q: str = "") -> list[dict]:
    """按时间倒序列出。q 非空时在标题、转写、纪要里做大小写不敏感的子串检索。"""
    if not MEETINGS_DIR.is_dir():
        return []
    needle = (q or "").strip().lower()
    out: list[dict] = []
    for d in sorted(MEETINGS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not _safe_id(d.name):
            continue
        m = Meeting(d.name, d)
        meta = m.meta()
        if not meta:
            continue
        if needle:
            hay = meta.get("title", "").lower()
            if needle not in hay:
                hay += " ".join(l.get("text", "") for l in m.transcript()).lower()
                hay += m.minutes_md().lower()
                if needle not in hay:
                    continue
        meta["has_audio"] = m.audio_path.exists()
        out.append(meta)
    return out


def remove(meeting_id: str) -> bool:
    m = get(meeting_id)
    if m is None:
        return False
    shutil.rmtree(m.dir, ignore_errors=True)
    return not m.dir.exists()


# ---------------------------------------------------------------- 导出

def fmt_ts(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    mm, ss = divmod(rem, 60)
    return f"{h}:{mm:02d}:{ss:02d}" if h else f"{mm:02d}:{ss:02d}"


def transcript_text(meeting: Meeting) -> str:
    """带时间戳的逐句转写，纯文本。"""
    meta = meeting.meta()
    head = [f"# {meta.get('title', meeting.id)}", ""]
    started = meta.get("started_at")
    if started:
        head.append(f"开始：{datetime.fromtimestamp(started):%Y-%m-%d %H:%M:%S}")
    head.append(f"时长：{fmt_ts(meta.get('duration'))}")
    head.append("")
    body = []
    for line in meeting.transcript():
        text = (line.get("text") or "").strip()
        if not text:
            continue
        spk = line.get("speaker")
        who = f"说话人 {spk} " if spk is not None and spk >= 0 else ""
        body.append(f"[{fmt_ts(line.get('beg'))}] {who}{text}")
    return "\n".join(head + (body or ["（无转写内容）"])) + "\n"


def export_bundle(meeting: Meeting) -> dict:
    """一次性导出全部内容，给 JSON 下载用。"""
    return {
        "meta": meeting.meta(),
        "transcript": meeting.transcript(),
        "minutes": meeting.minutes_json(),
        "minutes_markdown": meeting.minutes_md(),
    }
