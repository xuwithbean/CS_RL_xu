"""自动打开 Windows 游戏并实时传输到 Linux。

功能：
- 在 WSL 中通过 `powershell.exe` 启动 Windows 游戏进程。
- 在 Windows 中通过 `ffmpeg + gdigrab` 采集窗口/桌面并 UDP 推流到 Linux。
- 在 Linux 侧可选自动启动 `ffplay` 实时查看。
- Ctrl+C 时自动停止推流与播放器进程（不强制关闭游戏进程）。
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional


def run_tk_udp_viewer(
    port: int,
    ffmpeg_path: str = "ffmpeg",
    width: int = 1280,
    height: int = 720,
    window_title: str = "OpenGame Tk Viewer",
) -> None:
    """使用 Tkinter + ffmpeg 在 Linux 中显示 UDP 视频流。"""
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except Exception as e:
        raise RuntimeError(f"Tk viewer requires tkinter and Pillow: {e}")

    src = f"udp://0.0.0.0:{port}?overrun_nonfatal=1&fifo_size=5000000"
    frame_size = width * height * 3
    ffmpeg_cmd = [
        ffmpeg_path,
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-i",
        src,
        "-vf",
        f"scale={width}:{height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    root = tk.Tk()
    root.title(window_title)
    root.geometry(f"{width}x{height}")
    label = tk.Label(root, bg="black")
    label.pack(fill="both", expand=True)

    frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def reader() -> None:
        while not stop_event.is_set():
            if proc.stdout is None:
                break
            data = proc.stdout.read(frame_size)
            if not data or len(data) < frame_size:
                break
            try:
                while True:
                    frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(data)
            except queue.Full:
                pass

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    def refresh() -> None:
        try:
            frame = frame_queue.get_nowait()
            img = Image.frombytes("RGB", (width, height), frame)
            tkimg = ImageTk.PhotoImage(img)
            label.configure(image=tkimg)
            label.image = tkimg
        except queue.Empty:
            pass

        if proc.poll() is not None:
            try:
                err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            except Exception:
                err = ""
            if err.strip():
                print("[opengame-tk] decoder exited:")
                print(err.strip())
            root.after(500, root.destroy)
            return

        root.after(15, refresh)

    def on_close() -> None:
        stop_event.set()
        try:
            proc.terminate()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, refresh)
    root.mainloop()

    stop_event.set()
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=1)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class OpenGameTool:
    """Windows 游戏启动 + 实时推流工具。"""

    def __init__(
        self,
        game_exe: str,
        game_args: Optional[list[str]] = None,
        linux_ip: str = "auto",
        port: int = 1234,
        ffmpeg_path: str = "ffmpeg",
        ffplay_path: str = "ffplay",
        viewer_mode: str = "tk",
        web_port: int = 18080,
        framerate: int = 60,
        bitrate: str = "8M",
        window_title: Optional[str] = None,
        use_desktop_capture: bool = False,
    ):
        self.game_exe = game_exe
        self.game_args = game_args or []
        self.linux_ip = self._resolve_linux_ip(linux_ip)
        self.port = int(port)
        self.ffmpeg_path = ffmpeg_path
        self.ffplay_path = ffplay_path
        self.viewer_mode = viewer_mode
        self.web_port = int(web_port)
        self.framerate = int(framerate)
        self.bitrate = bitrate
        self.window_title = window_title
        self.use_desktop_capture = bool(use_desktop_capture)

        self.stream_dest = f"udp://{self.linux_ip}:{self.port}"
        self.viewer_src = f"udp://0.0.0.0:{self.port}"

        self._stream_proc: Optional[subprocess.Popen] = None
        self._viewer_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _list_window_titles(self) -> list[str]:
        """列出 Windows 当前可见窗口标题。"""
        ps = r'''
Add-Type -TypeDefinition @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class WinEnum {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);
}
"@ -Language CSharp;
$titles = New-Object System.Collections.Generic.List[string]
[WinEnum]::EnumWindows({
    param($hWnd, $lParam)
    if ([WinEnum]::IsWindowVisible($hWnd)) {
        $sb = New-Object System.Text.StringBuilder 512
        [void][WinEnum]::GetWindowText($hWnd, $sb, $sb.Capacity)
        $t = $sb.ToString().Trim()
        if ($t.Length -gt 0) { $titles.Add($t) }
    }
    return $true
}, [IntPtr]::Zero) | Out-Null
$titles | Sort-Object -Unique | ForEach-Object { Write-Output $_ }
'''
        try:
            proc = self._run_ps(ps, check=True)
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        except Exception:
            return []

    def _resolve_window_title(self) -> Optional[str]:
        """自动解析要抓取的窗口标题。

        规则：
        - 若显式传入标题且不是 `auto`，优先按该标题做子串匹配。
        - 否则按常见 CS/CS2 关键词匹配可见窗口。
        - 匹配不到返回 None，由上层决定是否退回桌面采集。
        """
        titles = self._list_window_titles()
        if not titles:
            return None

        wanted = []
        if self.window_title and self.window_title.lower() != "auto":
            wanted.append(self.window_title)
        wanted.extend([
            "Counter-Strike",
            "Counter Strike",
            "CS2",
            "反恐精英",
            "全球攻势",
        ])

        lowered = [(title, title.lower()) for title in titles]
        for key in wanted:
            key_lower = key.lower()
            for title, title_lower in lowered:
                if key_lower in title_lower:
                    return title
        return None

    @staticmethod
    def _resolve_linux_ip(linux_ip: str) -> str:
        """解析 Linux 接收地址。

        - `auto`：优先使用 `hostname -I` 的首个地址。
        - 解析失败时回退到 `127.0.0.1`。
        """
        if linux_ip != "auto":
            return linux_ip
        try:
            proc = subprocess.run(
                ["hostname", "-I"],
                check=True,
                capture_output=True,
                text=True,
            )
            parts = [item.strip() for item in proc.stdout.split() if item.strip()]
            if parts:
                return parts[0]
        except Exception:
            pass
        return "127.0.0.1"

    @staticmethod
    def _tail_err(pipe, max_bytes: int = 4000) -> str:
        """读取并截断子进程错误输出，便于快速定位问题。"""
        if pipe is None:
            return ""
        try:
            data = pipe.read()
            if not data:
                return ""
            if len(data) > max_bytes:
                return data[-max_bytes:]
            return data
        except Exception:
            return ""

    def _run_ps(self, script: str, check: bool = True) -> subprocess.CompletedProcess:
        """执行 PowerShell 脚本。"""
        return subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            check=check,
            capture_output=True,
            text=True,
        )

    def _build_start_game_ps(self) -> str:
        """构造启动游戏的 PowerShell 命令。"""
        exe = self.game_exe.replace("'", "''")
        if self.game_args:
            safe_args = []
            for a in self.game_args:
                safe_args.append("'" + a.replace("'", "''") + "'")
            arg_quoted = ",".join(safe_args)
            return f"Start-Process -FilePath '{exe}' -ArgumentList {arg_quoted}"
        return f"Start-Process -FilePath '{exe}'"

    def open_game(self, wait_seconds: float = 6.0) -> None:
        """打开 Windows 游戏进程。"""
        ps = self._build_start_game_ps()
        self._run_ps(ps, check=True)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _build_ffmpeg_cmd(self) -> list[str]:
        """构造 ffmpeg 推流命令。"""
        cmd = [
            self.ffmpeg_path,
            "-f",
            "gdigrab",
            "-framerate",
            str(self.framerate),
        ]

        if self.use_desktop_capture or not self.window_title or self.window_title.lower() == "auto":
            cmd += ["-i", "desktop"]
        else:
            cmd += ["-i", f"title={self.window_title}"]

        cmd += [
            "-vcodec",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            self.bitrate,
            "-f",
            "mpegts",
            self.stream_dest,
        ]
        return cmd

    def _build_ffplay_cmd(self) -> list[str]:
        """构造 ffplay 低延迟查看命令。"""
        return [
            self.ffplay_path,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-framedrop",
            "-strict",
            "-1",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            self.viewer_src,
        ]

    def _build_tk_viewer_cmd(self) -> list[str]:
        """构造 Tk 查看器子进程命令。"""
        return [
            sys.executable,
            os.path.abspath(__file__),
            "--viewer-only",
            "tk",
            "--port",
            str(self.port),
            "--ffmpeg",
            self.ffmpeg_path,
        ]

    def _build_web_viewer_cmd(self) -> list[str]:
        """构造 Web(MJPEG) 查看器命令（无 GUI 依赖）。"""
        src = f"udp://0.0.0.0:{self.port}?overrun_nonfatal=1&fifo_size=5000000"
        out = f"http://0.0.0.0:{self.web_port}/stream.mjpg"
        return [
            self.ffmpeg_path,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-i",
            src,
            "-f",
            "mpjpeg",
            "-listen",
            "1",
            out,
        ]

    def start_stream(self, restart_if_running: bool = False) -> Optional[subprocess.Popen]:
        """启动实时推流（Windows -> Linux UDP）。"""
        with self._lock:
            if self._stream_proc is not None:
                if not restart_if_running:
                    return self._stream_proc
                self.stop_stream()

            if not self.use_desktop_capture:
                resolved_title = self._resolve_window_title()
                if resolved_title:
                    if resolved_title != self.window_title:
                        print(f"[opengame] resolved window title -> {resolved_title}")
                    self.window_title = resolved_title
                elif self.window_title and self.window_title.lower() != "auto":
                    print(f"[opengame] requested title not found -> {self.window_title}")
                else:
                    # `auto` 未匹配到时，不再把它当作真实标题，直接走 desktop 采集
                    self.window_title = None

            cmd = self._build_ffmpeg_cmd()
            safe_cmd = " ".join(subprocess.list2cmdline([c]) for c in cmd)
            ps_cmd = f"& {{ {safe_cmd} }}"
            p = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 快速健康检查：若窗口采集失败，通常会在 1 秒内退出
            time.sleep(1.0)
            rc = p.poll()
            if rc is not None:
                err = self._tail_err(p.stderr)
                # 尝试自动回退：窗口标题采集失败时切到 desktop 采集
                if not self.use_desktop_capture:
                    print(f"[opengame] window capture failed (title={self.window_title}), fallback to desktop capture")
                    if err:
                        print("[opengame] ffmpeg error:")
                        print(err.strip())
                    self.use_desktop_capture = True
                    cmd2 = self._build_ffmpeg_cmd()
                    safe_cmd2 = " ".join(subprocess.list2cmdline([c]) for c in cmd2)
                    ps_cmd2 = f"& {{ {safe_cmd2} }}"
                    p2 = subprocess.Popen(
                        ["powershell.exe", "-NoProfile", "-Command", ps_cmd2],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    time.sleep(1.0)
                    if p2.poll() is None:
                        self._stream_proc = p2
                        print("[opengame] stream started with desktop capture")
                        return p2
                    err2 = self._tail_err(p2.stderr)
                    if err2:
                        print("[opengame] ffmpeg fallback error:")
                        print(err2.strip())
                    self._stream_proc = p2
                    return p2

                if err:
                    print("[opengame] ffmpeg exited early:")
                    print(err.strip())
            else:
                print(f"[opengame] stream started -> {self.stream_dest}")

            self._stream_proc = p
            return p

    def stop_stream(self) -> None:
        """停止推流进程。"""
        with self._lock:
            p = self._stream_proc
            self._stream_proc = None

        if p is None:
            return

        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=1)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def start_viewer(self, restart_if_running: bool = False) -> Optional[subprocess.Popen]:
        """在 Linux 侧启动查看器。"""
        with self._lock:
            if self._viewer_proc is not None:
                if not restart_if_running:
                    return self._viewer_proc
                self.stop_viewer()

            requested_mode = self.viewer_mode
            if self.viewer_mode == "tk":
                cmd = self._build_tk_viewer_cmd()
            elif self.viewer_mode == "web":
                cmd = self._build_web_viewer_cmd()
            else:
                cmd = self._build_ffplay_cmd()
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            time.sleep(0.8)
            rc = p.poll()
            if rc is not None:
                err = self._tail_err(p.stderr)
                print(f"[opengame] {self.viewer_mode} viewer exited early, viewer not shown")
                if err:
                    print(f"[opengame] {self.viewer_mode} viewer error:")
                    print(err.strip())
                if self.viewer_mode == "ffplay":
                    print(f"[opengame] hint: try manual test -> ffplay {self.viewer_src}")
                elif requested_mode == "tk":
                    print("[opengame] fallback: try ffplay viewer")
                    cmd = self._build_ffplay_cmd()
                    p2 = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                    time.sleep(0.8)
                    if p2.poll() is None:
                        self._viewer_proc = p2
                        self.viewer_mode = "ffplay"
                        print(f"[opengame] viewer started (ffplay fallback) <- {self.viewer_src}")
                        return p2
                    err2 = self._tail_err(p2.stderr)
                    print("[opengame] ffplay fallback also failed")
                    if err2:
                        print("[opengame] ffplay fallback error:")
                        print(err2.strip())
                    print("[opengame] Linux GUI display is likely unavailable. You need a working X/WSLg display to show video in Linux.")
            else:
                print(f"[opengame] viewer started ({self.viewer_mode}) <- {self.viewer_src}")
                if self.viewer_mode == "web":
                    print(f"[opengame] web viewer url: http://127.0.0.1:{self.web_port}/stream.mjpg")
                    print(f"[opengame] web viewer url (WSL IP): http://{self.linux_ip}:{self.web_port}/stream.mjpg")
            self._viewer_proc = p
            return p

    def stop_viewer(self) -> None:
        """停止 Linux 侧查看器。"""
        with self._lock:
            p = self._viewer_proc
            self._viewer_proc = None

        if p is None:
            return

        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=1)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    def start_all(self, wait_game_seconds: float = 6.0, start_viewer: bool = True) -> None:
        """先启动游戏，再启动推流，并可选启动 Linux 查看器。"""
        self.open_game(wait_seconds=wait_game_seconds)
        self.start_stream(restart_if_running=True)
        if start_viewer:
            # 给推流一个很短的准备时间
            time.sleep(0.2)
            self.start_viewer(restart_if_running=True)

    def stop_all(self) -> None:
        """停止查看器与推流（不强制结束游戏进程）。"""
        self.stop_viewer()
        self.stop_stream()

    def run_forever(self, wait_game_seconds: float = 6.0, start_viewer: bool = True) -> None:
        """一键启动并保持运行，直到 Ctrl+C。"""
        self.start_all(wait_game_seconds=wait_game_seconds, start_viewer=start_viewer)
        print(f"Streaming to udp://{self.linux_ip}:{self.port} (viewer={'on' if start_viewer else 'off'}, mode={self.viewer_mode})")
        if self.linux_ip == "127.0.0.1":
            print("[opengame] note: 若无画面可尝试将 --linux-ip 改为 WSL 实际 IP（例如 `hostname -I` 的地址）")
        print("Press Ctrl+C to stop stream/viewer")
        try:
            while True:
                # 运行时健康检查，避免静默失败
                if self._stream_proc is not None and self._stream_proc.poll() is not None:
                    err = self._tail_err(self._stream_proc.stderr)
                    print("[opengame] stream process exited")
                    if err:
                        print("[opengame] stream stderr:")
                        print(err.strip())
                    break
                if self._viewer_proc is not None and self._viewer_proc.poll() is not None:
                    err = self._tail_err(self._viewer_proc.stderr)
                    print("[opengame] viewer process exited")
                    if err:
                        print("[opengame] viewer stderr:")
                        print(err.strip())
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()

    def __enter__(self) -> "OpenGameTool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_all()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Open game on Windows and stream to Linux")
    p.add_argument("--viewer-only", choices=["tk"], default=None, help="Internal mode: run viewer only")
    p.add_argument("--game-exe", required=False, help="Windows game executable path")
    p.add_argument("--game-arg", action="append", default=[], help="Game argument, repeatable")
    p.add_argument("--linux-ip", default="auto", help="Linux receiver IP used by Windows ffmpeg, default auto")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--window-title", default="auto", help="Window title for gdigrab, default auto")
    p.add_argument("--desktop", action="store_true", help="Capture desktop instead of window title")
    p.add_argument("--framerate", type=int, default=60)
    p.add_argument("--bitrate", default="8M")
    p.add_argument("--wait-game", type=float, default=6.0, help="Seconds to wait after opening game")
    p.add_argument("--no-viewer", action="store_true", help="Do not start ffplay on Linux")
    p.add_argument("--viewer-mode", choices=["tk", "ffplay", "web"], default="tk")
    p.add_argument("--web-port", type=int, default=18080, help="HTTP port for web viewer mode")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffplay", default="ffplay")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.viewer_only == "tk":
        run_tk_udp_viewer(port=args.port, ffmpeg_path=args.ffmpeg)
        return

    if not args.game_exe:
        raise SystemExit("--game-exe is required unless --viewer-only tk is used")

    tool = OpenGameTool(
        game_exe=args.game_exe,
        game_args=args.game_arg,
        linux_ip=args.linux_ip,
        port=args.port,
        ffmpeg_path=args.ffmpeg,
        ffplay_path=args.ffplay,
        viewer_mode=args.viewer_mode,
        web_port=args.web_port,
        framerate=args.framerate,
        bitrate=args.bitrate,
        window_title=args.window_title,
        use_desktop_capture=args.desktop,
    )
    tool.run_forever(wait_game_seconds=args.wait_game, start_viewer=not args.no_viewer)


if __name__ == "__main__":
    main()
