"""桌面应用（PySide6）。

    python desktop.py

编排逻辑全在 engine.py，这里只负责界面。网页控制台 app.py 用的是同一个
engine.Engine，区别只在注入的 sink：那边推 WebSocket，这里发 Qt 信号。

线程模型：engine 是 asyncio 的，Qt 是自己的事件循环，两者不能混。
所以 asyncio 跑在独立线程里（EngineThread），跨线程通信一律走 Qt 信号——
Signal.emit 是线程安全的，连到主线程的槽会自动排队执行。
反方向（界面 -> engine）走 run_coroutine_threadsafe。
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (QObject, QThread, QTimer, QUrl, Qt, Signal, Slot)
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSlider, QSplitter,
    QStatusBar, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

import audio
import engine as eng
import meetings as mt

ROOT = Path(__file__).parent


# ---------------------------------------------------------------- 引擎线程

class QtSink(QObject):
    """engine 的事件出口。把 engine 的 async 调用转成 Qt 信号。"""

    status = Signal(dict)
    transcript = Signal(dict)
    minutes = Signal(dict)
    logged = Signal(dict)

    async def broadcast(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "status":
            self.status.emit(event)
        elif kind == "transcript":
            self.transcript.emit(event)
        elif kind == "minutes":
            self.minutes.emit(event)

    async def log(self, text: str, level: str = "info") -> None:
        self.logged.emit({"ts": time.strftime("%H:%M:%S"),
                          "text": text, "level": level})


class EngineThread(QThread):
    """把 asyncio 事件循环搬到独立线程，界面线程不被 IO 阻塞。"""

    failed = Signal(str)

    def __init__(self, sink: QtSink):
        super().__init__()
        self.sink = sink
        self.loop: asyncio.AbstractEventLoop | None = None
        self.engine: eng.Engine | None = None
        self.ready = threading.Event()

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.engine = eng.Engine(self.sink)
        except Exception as exc:            # 配置文件缺失/损坏
            self.failed.emit(str(exc))
            self.ready.set()
            return
        self.ready.set()
        self.loop.run_forever()

    def submit(self, coro) -> None:
        """从界面线程安全地把协程丢进引擎线程。"""
        if self.loop is None or not self.loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self) -> None:
        if self.loop is None or not self.loop.is_running():
            return
        # 先把会议正常封档，再停循环，否则录音的 WAV 头不会回填
        fut = asyncio.run_coroutine_threadsafe(self.engine.stop(), self.loop)
        try:
            fut.result(timeout=20)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.wait(5000)


# ---------------------------------------------------------------- 小部件

def dot(color: str) -> QPixmap:
    pm = QPixmap(10, 10)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(0, 0, 9, 9)
    p.end()
    return pm


DOT_COLORS = {"running": "#22c55e", "starting": "#f59e0b", "stopped": "#6b7280"}


class StatusLight(QWidget):
    def __init__(self, text: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(5)
        self.icon = QLabel()
        self.icon.setPixmap(dot(DOT_COLORS["stopped"]))
        self.label = QLabel(text)
        self.label.setStyleSheet("color: palette(mid);")
        lay.addWidget(self.icon)
        lay.addWidget(self.label)

    def set_state(self, state: str) -> None:
        self.icon.setPixmap(dot(DOT_COLORS.get(state, DOT_COLORS["stopped"])))


def minutes_html(data: dict) -> str:
    if not data or not data.get("total"):
        return "<p style='color:gray'>纪要会在这里持续更新</p>"
    out = []
    if data.get("overview"):
        out.append(
            "<div style='background:rgba(128,128,128,.12);border-left:3px solid #3b82f6;"
            f"padding:8px 10px;margin-bottom:14px'>{esc(data['overview'])}</div>")
    for g in data.get("groups", []):
        if not g["items"]:
            continue
        out.append(f"<div style='color:gray;font-size:12px;margin:12px 0 4px'>{g['label']}</div><ul style='margin:0'>")
        for it in g["items"]:
            owner = (f"<b style='color:#3b82f6'>@{esc(it['owner'])}</b> "
                     if it.get("owner") else "")
            star = " ⭐" if it.get("hits", 0) >= 3 else ""
            out.append(f"<li>{owner}{esc(it['text'])}{star}</li>")
        out.append("</ul>")
    return "".join(out)


def esc(s) -> str:
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def line_label(l: dict) -> str:
    spk = l.get("speaker")
    who = f"说话人 {spk}  " if spk is not None and spk >= 0 else ""
    return f"{mt.fmt_ts(l.get('beg')):>6}  {who}{l.get('text', '')}"


# ---------------------------------------------------------------- 历史窗口

class HistoryDialog(QDialog):
    """归档浏览：检索、回放、导出、改名、删除。

    双击转写里的某一句会把录音跳到那个时间点——这是做桌面版最想要的东西，
    浏览器里没法做得这么顺手。
    """

    def __init__(self, parent=None, running_id: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("历史会议")
        self.resize(1080, 680)
        self.running_id = running_id
        self.current: mt.Meeting | None = None
        self.detail: dict | None = None

        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self.on_position)
        self.player.durationChanged.connect(
            lambda d: self.seek.setRange(0, max(0, d)))
        self.player.playbackStateChanged.connect(self.on_play_state)
        # setSource 是异步的，媒体没加载完时 setPosition 会被静默丢掉。
        # 刚点开一场会议就双击某句的话跳转会失效，所以先记下来等加载完再跳。
        self._pending_seek: int | None = None
        self.player.mediaStatusChanged.connect(self.on_media_status)

        root = QHBoxLayout(self)

        # --- 左：检索 + 列表
        left = QVBoxLayout()
        self.query = QLineEdit(placeholderText="搜标题、转写原文、纪要内容…")
        self.query.textChanged.connect(self.debounce_search)
        self.count = QLabel()
        self.count.setStyleSheet("color: palette(mid); font-size: 11px;")
        self.listing = QListWidget()
        self.listing.currentItemChanged.connect(self.on_pick)
        left.addWidget(self.query)
        left.addWidget(self.count)
        left.addWidget(self.listing, 1)
        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(330)
        root.addWidget(left_box)

        # --- 右：详情
        right = QVBoxLayout()
        self.title = QLabel()
        self.title.setFont(QFont("", 14, QFont.Bold))
        self.title.setWordWrap(True)
        self.meta = QLabel()
        self.meta.setStyleSheet("color: palette(mid); font-size: 11px;")
        right.addWidget(self.title)
        right.addWidget(self.meta)

        bar = QHBoxLayout()
        for text, slot in (("导出纪要 .md", lambda: self.export("md")),
                           ("导出转写 .txt", lambda: self.export("txt")),
                           ("导出全部 .json", lambda: self.export("json")),
                           ("改标题", self.rename)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        self.btn_del = QPushButton("删除")
        self.btn_del.clicked.connect(self.remove)
        bar.addWidget(self.btn_del)
        right.addLayout(bar)

        play = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.clicked.connect(self.toggle_play)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.sliderMoved.connect(self.player.setPosition)
        self.clock = QLabel("00:00 / 00:00")
        self.clock.setStyleSheet("color: palette(mid); font-size: 11px;")
        play.addWidget(self.btn_play)
        play.addWidget(self.seek, 1)
        play.addWidget(self.clock)
        self.play_row = QWidget()
        self.play_row.setLayout(play)
        right.addWidget(self.play_row)

        self.tabs = QTabWidget()
        self.view_minutes = QTextBrowser()
        self.view_lines = QListWidget()
        self.view_lines.setFont(QFont("Consolas", 10))
        self.view_lines.itemDoubleClicked.connect(self.seek_to_line)
        self.tabs.addTab(self.view_minutes, "纪要")
        self.tabs.addTab(self.view_lines, "转写原文（双击某句跳到录音处）")
        right.addWidget(self.tabs, 1)
        root.addLayout(right, 1)

        self._timer = QTimer(self, singleShot=True, interval=250)
        self._timer.timeout.connect(self.reload)
        self.reload()

    # ---- 列表

    def debounce_search(self) -> None:
        self._timer.start()

    def reload(self) -> None:
        q = self.query.text().strip()
        rows = mt.listing(q)
        self.count.setText(f"{len(rows)} 场匹配" if q else f"共 {len(rows)} 场")
        keep = self.current.id if self.current else None
        self.listing.blockSignals(True)
        self.listing.clear()
        for r in rows:
            started = datetime.fromtimestamp(r["started_at"]) if r.get("started_at") else None
            sub = (f"{started:%Y-%m-%d %H:%M}" if started else "") + \
                  f" · {mt.fmt_ts(r.get('duration'))} · {r.get('line_count', 0)} 句" \
                  f" · {r.get('item_count', 0)} 条" + (" · 🎧" if r.get("has_audio") else "")
            it = QListWidgetItem(f"{r['title']}\n{sub}")
            it.setData(Qt.UserRole, r["id"])
            self.listing.addItem(it)
        self.listing.blockSignals(False)
        if keep:
            for i in range(self.listing.count()):
                if self.listing.item(i).data(Qt.UserRole) == keep:
                    self.listing.setCurrentRow(i)
                    return
        if not rows:
            self.show_detail(None)

    def on_pick(self, item: QListWidgetItem | None, _prev=None) -> None:
        if item is None:
            return
        self.show_detail(mt.get(item.data(Qt.UserRole)))

    # ---- 详情

    def show_detail(self, meeting: mt.Meeting | None) -> None:
        self.player.stop()
        self._pending_seek = None
        self.current = meeting
        if meeting is None:
            self.title.setText("")
            self.meta.setText("左边选一场会议")
            self.view_minutes.setHtml("")
            self.view_lines.clear()
            self.play_row.setVisible(False)
            return
        mt.repair_wav(meeting.audio_path)       # 上次被强杀的话头是坏的
        meta = meeting.meta()
        started = datetime.fromtimestamp(meta["started_at"]) if meta.get("started_at") else None
        self.title.setText(meta.get("title", meeting.id))
        self.meta.setText(
            (f"{started:%Y-%m-%d %H:%M}  ·  " if started else "")
            + f"时长 {mt.fmt_ts(meta.get('duration'))}"
            + f"  ·  录音 {mt.fmt_ts(meta.get('audio_seconds'))}"
            + f"  ·  {meta.get('line_count', 0)} 句"
            + f"  ·  {meta.get('item_count', 0)} 条纪要"
            + ("   [正在进行]" if meeting.id == self.running_id else ""))
        self.view_minutes.setHtml(minutes_html(meeting.minutes_json()))
        self.view_lines.clear()
        for l in meeting.transcript():
            if not (l.get("text") or "").strip():
                continue
            it = QListWidgetItem(line_label(l))
            it.setData(Qt.UserRole, l.get("beg"))
            self.view_lines.addItem(it)
        has_audio = meeting.audio_path.exists()
        self.play_row.setVisible(has_audio)
        if has_audio:
            self.player.setSource(QUrl.fromLocalFile(str(meeting.audio_path)))

    # ---- 回放

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_play_state(self, state) -> None:
        self.btn_play.setText("⏸" if state == QMediaPlayer.PlayingState else "▶")

    def on_position(self, pos: int) -> None:
        if not self.seek.isSliderDown():
            self.seek.setValue(pos)
        self.clock.setText(
            f"{mt.fmt_ts(pos / 1000)} / {mt.fmt_ts(self.player.duration() / 1000)}")

    LOADED = (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia)

    def on_media_status(self, status) -> None:
        if status in self.LOADED and self._pending_seek is not None:
            self.player.setPosition(self._pending_seek)
            self._pending_seek = None

    def seek_to_line(self, item: QListWidgetItem) -> None:
        beg = item.data(Qt.UserRole)
        if beg is None or not self.current or not self.current.audio_path.exists():
            return
        pos = int(beg * 1000)
        if self.player.mediaStatus() in self.LOADED:
            self.player.setPosition(pos)
        else:
            self._pending_seek = pos
        self.player.play()

    # ---- 操作

    def export(self, fmt: str) -> None:
        if self.current is None:
            return
        title = self.current.meta().get("title") or self.current.id
        safe = "".join(c for c in title if c not in '\\/:*?"<>|').strip() or self.current.id
        name, body = {
            "md": (f"{safe}-纪要.md", self.current.minutes_md()),
            "txt": (f"{safe}-转写.txt", mt.transcript_text(self.current)),
            "json": (f"{safe}-全部.json",
                     json.dumps(mt.export_bundle(self.current), ensure_ascii=False, indent=2)),
        }[fmt]
        path, _ = QFileDialog.getSaveFileName(self, "导出", str(Path.home() / name))
        if not path:
            return
        try:
            Path(path).write_text(body, encoding="utf-8", newline="\n")
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def rename(self) -> None:
        if self.current is None:
            return
        old = self.current.meta().get("title", "")
        text, ok = QInputDialog.getText(self, "会议标题", "标题", text=old)
        if ok and text.strip():
            self.current.write_meta(title=text.strip()[:120])
            self.show_detail(self.current)
            self.reload()

    def remove(self) -> None:
        if self.current is None:
            return
        if self.current.id == self.running_id:
            QMessageBox.information(self, "删不了", "这场会议正在进行，先停止再删。")
            return
        title = self.current.meta().get("title", self.current.id)
        if QMessageBox.question(
                self, "删除会议",
                f"删除「{title}」？\n录音、转写、纪要都会一起删掉，无法恢复。"
        ) != QMessageBox.Yes:
            return
        self.player.setSource(QUrl())       # 松开文件句柄，否则 Windows 删不掉
        mt.remove(self.current.id)
        self.current = None
        self.reload()
        self.show_detail(None)

    def closeEvent(self, e) -> None:
        self.player.stop()
        super().closeEvent(e)


# ---------------------------------------------------------------- 主窗口

class MainWindow(QMainWindow):
    def __init__(self, thread: EngineThread, sink: QtSink):
        super().__init__()
        self.thread_ = thread
        self.setWindowTitle("实时会议纪要")
        self.resize(1280, 820)
        self.state = "idle"
        self.meeting_id: str | None = None

        # --- 顶栏
        top = QHBoxLayout()
        self.light_llm = StatusLight("摘要模型")
        self.light_asr = StatusLight("转写服务")
        self.light_cap = StatusLight("采集")
        for w in (self.light_llm, self.light_asr, self.light_cap):
            top.addWidget(w)

        top.addWidget(QLabel("声音来源"))
        self.device = QComboBox()
        self.device.setMinimumWidth(340)
        self.device.currentIndexChanged.connect(self.on_device)
        top.addWidget(self.device)

        self.chk_loopback = QCheckBox("采系统声音")
        self.chk_loopback.stateChanged.connect(
            lambda: self.patch(loopback=self.chk_loopback.isChecked()))
        self.chk_gain = QCheckBox("自动增益")
        self.chk_gain.stateChanged.connect(
            lambda: self.patch(auto_gain=self.chk_gain.isChecked()))
        top.addWidget(self.chk_loopback)
        top.addWidget(self.chk_gain)
        top.addStretch(1)

        self.rec = QLabel()
        top.addWidget(self.rec)
        self.btn_history = QPushButton("历史")
        self.btn_history.clicked.connect(self.open_history)
        self.btn_start = QPushButton("开始")
        self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        for b in (self.btn_history, self.btn_start, self.btn_stop):
            top.addWidget(b)

        # --- 主体
        self.lines = QListWidget()
        self.lines.setFont(QFont("Consolas", 10))
        self.view_minutes = QTextBrowser()
        left = self.titled("实时转写", self.lines)
        right = self.titled("滚动纪要", self.view_minutes)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([640, 640])

        self.logs = QPlainTextEdit(readOnly=True)
        self.logs.setFont(QFont("Consolas", 9))
        self.logs.setMaximumBlockCount(500)
        self.logs.setFixedHeight(150)

        body = QVBoxLayout()
        body.addLayout(top)
        body.addWidget(split, 1)
        body.addWidget(self.titled("日志", self.logs))
        holder = QWidget()
        holder.setLayout(body)
        self.setCentralWidget(holder)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

        sink.status.connect(self.on_status)
        sink.transcript.connect(self.on_transcript)
        sink.minutes.connect(self.on_minutes)
        sink.logged.connect(self.on_log)

        self.load_devices()
        self.on_minutes({})

    @staticmethod
    def titled(text: str, inner: QWidget) -> QWidget:
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        lab = QLabel(text)
        lab.setStyleSheet("color: palette(mid); font-size: 11px; padding: 2px;")
        box.addWidget(lab)
        box.addWidget(inner, 1)
        w = QWidget()
        w.setLayout(box)
        return w

    # ---- 配置

    def cfg(self) -> dict:
        return self.thread_.engine.cfg

    def load_devices(self) -> None:
        try:
            devices = audio.list_devices()
        except Exception as exc:
            self.on_log({"ts": time.strftime("%H:%M:%S"),
                         "text": f"枚举音频设备失败：{exc}", "level": "error"})
            devices = []
        conf = self.cfg()["audio"]
        self.device.blockSignals(True)
        self.device.clear()
        for d in sorted(devices, key=lambda d: not d["loopback_candidate"]):
            tag = "🔊" if d["loopback_candidate"] else "🎤"
            self.device.addItem(
                f"{tag} {d['name']} — {d['rate']}Hz {d['channels']}ch", d["index"])
        idx = self.device.findData(conf.get("device"))
        if idx >= 0:
            self.device.setCurrentIndex(idx)
        self.device.blockSignals(False)
        self.chk_loopback.setChecked(bool(conf.get("loopback", True)))
        self.chk_gain.setChecked(conf.get("auto_gain", True) is not False)

    def on_device(self) -> None:
        data = self.device.currentData()
        if data is not None:
            self.patch(device=int(data))

    def patch(self, **kw) -> None:
        if self.thread_.engine is None:
            return
        self.cfg()["audio"].update(kw)
        eng.save_config(self.cfg())

    # ---- 引擎事件

    @Slot(dict)
    def on_status(self, msg: dict) -> None:
        self.state = msg.get("state", "idle")
        svc = msg.get("services") or {}
        fallback = "starting" if self.state == "starting" else "stopped"
        self.light_llm.set_state(svc.get("llm", fallback))
        self.light_asr.set_state(svc.get("asr", fallback))
        self.light_cap.set_state(
            "running" if self.state == "running" else fallback)
        self.btn_start.setEnabled(self.state == "idle")
        self.btn_stop.setEnabled(self.state != "idle")
        self.btn_start.setText("启动中…" if self.state == "starting" else "开始")
        m = msg.get("meeting")
        self.meeting_id = m["id"] if m else None
        self.rec.setText(f"<span style='color:#ef4444'>● 录制中</span>  {m['title']}"
                         if m else "")
        self.statusBar().showMessage(
            f"归档到 meetings/{m['id']}/" if m else "就绪")

    @Slot(dict)
    def on_transcript(self, msg: dict) -> None:
        bar = self.lines.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 24
        self.lines.clear()
        for l in msg.get("lines", []):
            self.lines.addItem(line_label(l))
        if msg.get("buffer"):
            it = QListWidgetItem(f"{'':>6}  {msg['buffer']}")
            it.setForeground(QColor("#9ca3af"))
            self.lines.addItem(it)
        if at_bottom:
            self.lines.scrollToBottom()

    @Slot(dict)
    def on_minutes(self, msg: dict) -> None:
        self.view_minutes.setHtml(minutes_html(msg))

    @Slot(dict)
    def on_log(self, msg: dict) -> None:
        line = f"{msg['ts']}  {msg['text']}"
        if msg.get("level") == "error":
            self.logs.appendHtml(
                f"<span style='color:#ef4444'>{esc(line)}</span>")
        else:
            self.logs.appendPlainText(line)

    # ---- 动作

    def start(self) -> None:
        self.btn_start.setEnabled(False)
        self.lines.clear()
        self.thread_.submit(self.thread_.engine.start())

    def stop(self) -> None:
        self.btn_stop.setEnabled(False)
        self.thread_.submit(self.thread_.engine.stop())

    def open_history(self) -> None:
        HistoryDialog(self, running_id=self.meeting_id).exec()

    def closeEvent(self, e) -> None:
        if self.state != "idle":
            if QMessageBox.question(
                    self, "还在录制",
                    "会议正在进行，退出会先停止并封档。确定退出？") != QMessageBox.Yes:
                e.ignore()
                return
        self.statusBar().showMessage("正在停止并封档…")
        QApplication.processEvents()
        self.thread_.shutdown()
        super().closeEvent(e)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("实时会议纪要")
    if sys.platform == "win32":
        app.setFont(QFont("Microsoft YaHei UI", 9))

    sink = QtSink()
    thread = EngineThread(sink)
    thread.start()
    if not thread.ready.wait(10) or thread.engine is None:
        QMessageBox.critical(None, "启动失败",
                             "引擎没能初始化，多半是 config.json 缺失或损坏。")
        return 1

    win = MainWindow(thread, sink)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
