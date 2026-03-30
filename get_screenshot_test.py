"""OpenGameTool 端到端示例：窗口推流 + 定时截图。

运行：
  python get_screenshot_test.py 20
"""

from __future__ import annotations

import re
import sys
import time

from opengame import OpenGameTool


def parse_udp_endpoint(url: str) -> tuple[str, int]:
    m = re.match(r"^udp://([^:/?#]+):(\d+)", (url or "").strip())
    if not m:
        raise ValueError(f"invalid udp endpoint: {url}")
    return m.group(1), int(m.group(2))


def main(run_seconds: int = 20, dest: str = "udp://192.168.221.36:1234") -> None:
    ip, port = parse_udp_endpoint(dest)
    tool = OpenGameTool(
        game_exe=r"E:\steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe",
        game_args=["-applaunch", "730"],
        linux_ip=ip,
        port=port,
        framerate=30,
        bitrate="1500k",
        window_title="auto",
        stream_outputs=[dest],
    )

    print(f"Starting stream -> {dest}")
    try:
        tool.open_game(wait_seconds=6.0)
        p = tool.start_stream(with_viewer=False)
        if p is None:
            print("Failed to start stream")
            return

        print(f"Running for {run_seconds}s...")
        start_t = time.time()
        shot_done = False
        while time.time() - start_t < run_seconds:
            elapsed = time.time() - start_t
            if not shot_done and elapsed >= min(5, run_seconds):
                shot_done = True
                tool.capture_screenshot("screenshots/test_capture.jpg")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Stopping stream...")
        tool.stop_stream()


if __name__ == "__main__":
    secs = 20
    if len(sys.argv) > 1:
        try:
            secs = int(sys.argv[1])
        except Exception:
            pass
    main(run_seconds=secs)

