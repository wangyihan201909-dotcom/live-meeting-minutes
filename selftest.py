"""自检：不依赖任何服务，单独验证音频采集这一环。

    python selftest.py

打开 WASAPI loopback 流采 3 秒，报告采样格式、字节数和音量。
建议运行时放点声音（音乐、视频都行），否则只能验证流能打开、采不到内容。
"""

from __future__ import annotations

import time

import numpy as np

import audio


def main() -> int:
    devs = audio.list_devices()
    good = [d for d in devs if d["loopback_candidate"]]

    print(f"共 {len(devs)} 个输入设备，其中可采系统声音的 {len(good)} 个：")
    for d in good:
        print(f"  [{d['index']:2d}] {d['name']}  {d['rate']}Hz {d['channels']}ch")
    if not good:
        print("\n没有可用的 WASAPI loopback 设备。")
        print("装一个 VB-Audio Virtual Cable，把系统默认播放设备设成它。")
        return 1

    chunks: list[bytes] = []
    print("\n采集 3 秒…（现在放点声音）")
    try:
        with audio.Capture(None, chunks.append) as cap:
            print(f"设备 {cap.index}：{cap.rate}Hz {cap.channels}ch → 16000Hz 1ch")
            time.sleep(3.0)
    except Exception as exc:
        print(f"\n打开失败：{exc}")
        return 1

    total = sum(len(c) for c in chunks)
    seconds = total / 2 / audio.TARGET_RATE
    print(f"\n采到 {len(chunks)} 块，{total} 字节，转换后约 {seconds:.2f} 秒（应接近 3 秒）")

    if not total:
        print("流打开了但一个字节都没采到。")
        return 1

    x = np.frombuffer(b"".join(chunks), dtype=np.int16)
    peak = int(np.abs(x).max())
    rms = float(np.sqrt((x.astype(np.float32) ** 2).mean()))
    print(f"峰值 {peak}/32768（{peak / 32768 * 100:.1f}%）  RMS {rms:.0f}")

    if peak < 100:
        print("\n流是通的，但几乎是静音。放点声音再跑一次确认。")
        return 0
    print("\n音频采集完全正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
