"""端到端示例：在 Windows 启动 ffmpeg 将桌面2 推送到指定 udp 地址，并在 WSL 用 ffplay 播放。

运行前提：
- Windows 可被 WSL 调用 `powershell.exe`（默认），并且 Windows 上有 `ffmpeg.exe`。
- WSL 中有 `ffplay`（或 ffmpeg 能够接收并播放流）。

使用：
  python test.py 15
  （可选第一个参数为运行秒数，默认 15 秒）
"""

import sys
import time
import threading
from datetime import datetime
from get_screenshot import ScreenshotTool


def main(run_seconds: int = 15, dest: str = "udp://192.168.221.36:1234"):
	st = ScreenshotTool(
		ffmpeg_path="ffmpeg",
		ffplay_path="ffplay",
		ffmpeg_dest=dest,
		ffplay_source=dest,
		framerate=30,
		bitrate='500k',
		gop=6,   
	)
	print(f"Starting stream -> {dest} (capture monitor 2). Viewer will open with ffplay.")
	try:
		proc = st.start_stream_monitor(monitor=2, dest=dest, background=True)
		if proc is None:
			print("Failed to start stream on Windows side.")
			return
		# 启动 viewer（在 WSL）
		v = st.start_viewer(restart_if_running=True)
		if v is None:
			print("Viewer failed to start - check ffplay availability in WSL.")

		# 后台线程：将终端置为原始模式，监听单键（无需回车）。'l' 抓帧，'q' 退出。
		stop_event = threading.Event()

		def keyloop_raw():
			import tty
			import termios
			import select

			fd = sys.stdin.fileno()
			old_settings = termios.tcgetattr(fd)
			try:
				tty.setraw(fd)
				print("按键监听已启动：按 'l' 抓取一帧，按 'q' 退出。")
				while not stop_event.is_set():
					r, _, _ = select.select([sys.stdin], [], [], 0.1)
					if not r:
						continue
					ch = sys.stdin.read(1)
					if not ch:
						continue
					if ch == 'l':
						img = st.get_one_frame(monitor=2, return_pil=True)
						ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
						if img is None:
							print(f"[{ts}] 捕获失败")
							continue
						try:
							# 优先当作 PIL.Image
							if hasattr(img, 'save'):
								fname = f"frame_{ts}.png"
								img.save(fname)
							else:
								fname = f"frame_{ts}.png"
								with open(fname, 'wb') as f:
									f.write(img)
							print(f"[{ts}] 已保存: {fname}")
						except Exception as e:
							try:
								fname = f"frame_{ts}.png"
								with open(fname, 'wb') as f:
									f.write(img if isinstance(img, (bytes, bytearray)) else bytes(img))
								print(f"[{ts}] 已保存: {fname}")
							except Exception as e2:
								print(f"保存图片失败: {e2}")
					elif ch == 'q' or ch == '\x03':  # Ctrl-C 也当作退出
						stop_event.set()
						break
			finally:
				try:
					termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
				except Exception:
					pass

		t = threading.Thread(target=keyloop_raw, daemon=True)
		t.start()

		print(f"Running for {run_seconds} seconds... (press 'q' or Ctrl-C to quit)")
		start_t = time.time()
		try:
			while time.time() - start_t < run_seconds and not stop_event.is_set():
				time.sleep(0.1)
		finally:
			stop_event.set()
	except KeyboardInterrupt:
		print("Interrupted by user")
	finally:
		print("Stopping stream and viewer...")
		try:
			st.stop_end_to_end()
		except Exception:
			pass


if __name__ == '__main__':
	secs = 30
	if len(sys.argv) > 1:
		try:
			secs = int(sys.argv[1])
		except Exception:
			pass
	main(run_seconds=secs)

