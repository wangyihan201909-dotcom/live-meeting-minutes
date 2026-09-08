"""音频采集。

只采麦克风的话，纪要里就只有自己的发言，别人说什么全丢——所以默认走
WASAPI loopback，直接采系统正在播放的声音（会议软件的输出）。

用 PyAudioWPatch 而不是 sounddevice：sounddevice 0.5.6 的 WasapiSettings
只有 exclusive / auto_convert / explicit_sample_format，**不提供 loopback**。
PyAudioWPatch 是专门为 WASAPI loopback 打补丁的 PyAudio 分支。

loopback 设备只能按其原生格式采集（通常 48kHz 立体声），
而 WhisperLiveKit 的 --pcm-input 要 16kHz 单声道 s16le，所以本模块负责转换。
"""

from __future__ import annotations

import asyncio
import atexit
import threading

import numpy as np
import pyaudiowpatch as pyaudio

TARGET_RATE = 16000          # WhisperLiveKit --pcm-input 约定
BLOCK_SECONDS = 0.5

_pa: pyaudio.PyAudio | None = None
_lock = threading.Lock()


def _instance() -> pyaudio.PyAudio:
    """全进程共用一个 PyAudio。

    反复 PyAudio() / terminate() 会让 PortAudio 在 Windows WASAPI 上挂死——
    枚举设备再打开流就必然卡住，所以这里只初始化一次。
    """
    global _pa
    with _lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
            atexit.register(_shutdown)
        return _pa


def _shutdown() -> None:
    global _pa
    with _lock:
        if _pa is not None:
            _pa.terminate()
            _pa = None


def list_devices() -> list[dict]:
    """枚举设备。loopback_candidate 标出「能采到别人声音」的那些。"""
    p = _instance()
    loopback_indices = set()
    out: list[dict] = []
    for dev in p.get_loopback_device_info_generator():
        loopback_indices.add(dev["index"])
        out.append({
            "index": dev["index"],
            "name": dev["name"],
            "hostapi": "WASAPI Loopback",
            "rate": int(dev["defaultSampleRate"]),
            "channels": dev["maxInputChannels"],
            "loopback_candidate": True,
        })
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev["maxInputChannels"] <= 0 or i in loopback_indices:
            continue
        out.append({
            "index": i,
            "name": dev["name"],
            "hostapi": p.get_host_api_info_by_index(dev["hostApi"])["name"],
            "rate": int(dev["defaultSampleRate"]),
            "channels": dev["maxInputChannels"],
            "loopback_candidate": False,
        })
    return out


def default_loopback_device() -> int | None:
    """优先返回系统默认输出对应的 loopback 设备。"""
    p = _instance()
    try:
        api = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        name = p.get_device_info_by_index(api["defaultOutputDevice"])["name"]
    except Exception:
        name = None
    first = None
    for dev in p.get_loopback_device_info_generator():
        if first is None:
            first = dev["index"]
        if name and name in dev["name"]:
            return dev["index"]
    return first


def device_format(index: int | None) -> tuple[int, int, int]:
    """返回 (设备索引, 采样率, 声道数)。索引为 None 时自动挑 loopback。"""
    if index is None:
        index = default_loopback_device()
    if index is None:
        raise RuntimeError(
            "找不到可用于采集系统声音的 WASAPI loopback 设备；"
            "可安装 VB-Audio Virtual Cable 作为替代"
        )
    dev = _instance().get_device_info_by_index(index)
    return index, int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])


def to_16k_mono(raw: bytes, rate: int, channels: int) -> bytes:
    """降混单声道 + 重采样到 16kHz。整数倍降采样先做移动平均抗混叠。"""
    x = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        usable = len(x) - len(x) % channels
        x = x[:usable].reshape(-1, channels).mean(axis=1)
    x = x.astype(np.float32)

    if rate != TARGET_RATE and len(x):
        ratio = rate / TARGET_RATE
        if ratio == int(ratio) and len(x) >= int(ratio):
            step = int(ratio)
            kernel = np.ones(step, dtype=np.float32) / step
            x = np.convolve(x, kernel, mode="same")[::step]
        else:
            count = max(1, int(len(x) / ratio))
            x = np.interp(np.arange(count) * ratio, np.arange(len(x)), x)

    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()


class Capture:
    """打开采集流，把转换后的 16k 单声道 PCM 交给回调。"""

    def __init__(self, index: int | None, on_pcm):
        self.index, self.rate, self.channels = device_format(index)
        self.on_pcm = on_pcm
        self._stream = None

    def __enter__(self) -> "Capture":
        def callback(in_data, _count, _info, _status):
            self.on_pcm(to_16k_mono(in_data, self.rate, self.channels))
            return (None, pyaudio.paContinue)

        self._stream = _instance().open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.index,
            frames_per_buffer=int(self.rate * BLOCK_SECONDS),
            stream_callback=callback,
        )
        return self

    def __exit__(self, *_exc) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        # 不 terminate：PyAudio 实例全进程共用，退出时由 atexit 收尾


BLOCK_BYTES = int(TARGET_RATE * BLOCK_SECONDS) * 2      # 16000 采样 = 32000 字节


async def send_audio(ws, device: int | None, loopback: bool = True, on_log=None) -> None:
    """采集 PCM 并按墙钟节奏推给转写服务，静音期补零。

    WhisperLiveKit 每个 WS 连接是独立会话、不广播转写，
    所以必须由我们自己当那个唯一的客户端，不能另开浏览器采音频。

    补静音是必须的：WASAPI loopback 在系统没有播放声音时**完全不产生数据包**
    （实测 10 秒里只有前 1.8 秒有数据）。不补的话转写服务的时间轴不会推进，
    上一句的结尾会一直停在未确认状态，直到下次有人开口才被冲出来。
    """
    loop = asyncio.get_running_loop()
    buf = bytearray()
    lock = threading.Lock()
    silence = bytes(BLOCK_BYTES)

    def on_pcm(pcm: bytes) -> None:
        with lock:
            buf.extend(pcm)
            if len(buf) > BLOCK_BYTES * 20:   # 积压超过 10 秒就丢旧的，别无限涨
                del buf[: len(buf) - BLOCK_BYTES * 10]

    with Capture(device, on_pcm) as cap:
        if on_log:
            await on_log(
                f"开始采集：设备 {cap.index}，"
                f"{cap.rate}Hz {cap.channels}ch → {TARGET_RATE}Hz 单声道"
            )
        deadline = loop.time()
        while True:
            deadline += BLOCK_SECONDS
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            with lock:
                if len(buf) >= BLOCK_BYTES:
                    chunk = bytes(buf[:BLOCK_BYTES])
                    del buf[:BLOCK_BYTES]
                else:
                    chunk = bytes(buf) + silence[len(buf):]
                    buf.clear()
            await ws.send(chunk)
