"""自动打开 Windows 游戏并采集 CS 窗口画面。

功能：
- 在 WSL 中通过 powershell.exe 启动 Windows 游戏。
- 仅使用 gdigrab 的 title=... 方式抓取 CS 窗口并 UDP 推流。
- 可选在 Windows 侧启动 ffplay 预览（读取本机 UDP 流）。
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


class OpenGameTool:
    """Windows 游戏启动 + 窗口推流工具（简化版）。"""

    def __init__(
        self,
        game_exe: str,
        game_args: Optional[list[str]] = None,
        linux_ip: str = "auto",
        port: int = 12345,
        ffmpeg_path: str = "ffmpeg",
        ffplay_path: str = "ffplay",
        framerate: int = 60,
        bitrate: str = "8M",
        window_title: str = "auto",
        stream_outputs: Optional[list[str]] = None,
        viewer_source: Optional[str] = None,
        view_width: int = 800,
        view_height: int = 450,
    ):
        self.game_exe = game_exe
        self.game_args = game_args or []
        self.linux_ip = self._resolve_linux_ip(linux_ip)
        self.port = int(port)
        self.ffmpeg_path = ffmpeg_path
        self.ffplay_path = ffplay_path
        self.framerate = int(framerate)
        self.bitrate = bitrate
        self.window_title = window_title
        self.stream_outputs = [s for s in (stream_outputs or []) if s]
        self.view_width = int(view_width)
        self.view_height = int(view_height)

        self.stream_dest = f"udp://{self.linux_ip}:{self.port}"
        self.win_viewer_src = f"udp://127.0.0.1:{self.port}?fifo_size=1000000&overrun_nonfatal=1"
        self.viewer_source = viewer_source or self.win_viewer_src

        if not self.stream_outputs:
            self.stream_outputs = [self.stream_dest]

        self._stream_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._cmd_stop = threading.Event()
        self._cmd_thread: Optional[threading.Thread] = None
        self._shot_stop = threading.Event()
        self._shot_thread: Optional[threading.Thread] = None

    def _command_listener(self) -> None:
        """后台读取终端命令，转发到线程安全队列。"""
        while not self._cmd_stop.is_set():
            try:
                line = sys.stdin.readline()
                if line == "":
                    time.sleep(0.1)
                    continue
                cmd = line.strip().lower()
                if cmd:
                    self._cmd_queue.put(cmd)
            except Exception:
                time.sleep(0.1)

    def _start_command_listener(self) -> None:
        """启动终端命令监听线程。"""
        if self._cmd_thread is not None and self._cmd_thread.is_alive():
            return
        self._cmd_stop.clear()
        self._cmd_thread = threading.Thread(target=self._command_listener, daemon=True)
        self._cmd_thread.start()

    def _stop_command_listener(self) -> None:
        """停止终端命令监听线程。"""
        self._cmd_stop.set()

    def _handle_runtime_command(self, cmd: str) -> None:
        """处理运行时终端命令。"""
        if cmd == "screenshot":
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join("screenshots", f"screenshot_{ts}.jpg")
            print(f"[opengame] screenshot requested -> {out}")
            ok = self.capture_screenshot(out)
            if ok:
                print(f"[opengame] screenshot success -> {out}")
            else:
                print("[opengame] screenshot failed")
            return

        if cmd == "screenshot_100":
            self._start_batch_screenshot(total=100, interval_sec=0.1)
            return

        if cmd == "p":
            self._stop_batch_screenshot(wait=False)
            return

        print(f"[opengame] unknown command: {cmd}")
        print("[opengame] supported commands: screenshot | screenshot_100 | p")

    def _batch_screenshot_worker(self, total: int, interval_sec: float) -> None:
        """后台批量截图：固定间隔截图，可被停止。"""
        print(f"[opengame] screenshot_100 started: total={total}, interval={interval_sec:.1f}s")
        captured = 0
        try:
            for idx in range(1, total + 1):
                if self._shot_stop.is_set():
                    break

                ts = time.strftime("%Y%m%d_%H%M%S")
                out = os.path.join("screenshots", f"screenshot_100_{idx:03d}_{ts}.jpg")
                ok = self.capture_screenshot(out)
                if ok:
                    captured += 1
                    print(f"[opengame] screenshot_100 progress: {idx}/{total}")
                else:
                    print(f"[opengame] screenshot_100 failed at {idx}/{total}")

                if idx >= total:
                    break

                start_wait = time.monotonic()
                while (time.monotonic() - start_wait) < interval_sec:
                    if self._shot_stop.is_set():
                        break
                    time.sleep(0.1)
                if self._shot_stop.is_set():
                    break
        finally:
            stopped = self._shot_stop.is_set()
            self._shot_stop.clear()
            self._shot_thread = None
            if stopped:
                print(f"[opengame] screenshot_100 stopped by user, captured={captured}")
            else:
                print(f"[opengame] screenshot_100 finished, captured={captured}")

    def _start_batch_screenshot(self, total: int, interval_sec: float) -> None:
        """启动批量截图任务。"""
        if self._shot_thread is not None and self._shot_thread.is_alive():
            print("[opengame] screenshot_100 is already running; input 'p' to stop")
            return
        self._shot_stop.clear()
        self._shot_thread = threading.Thread(
            target=self._batch_screenshot_worker,
            args=(int(total), float(interval_sec)),
            daemon=True,
        )
        self._shot_thread.start()

    def _stop_batch_screenshot(self, wait: bool = False) -> None:
        """停止批量截图任务。"""
        if self._shot_thread is None or not self._shot_thread.is_alive():
            print("[opengame] no running screenshot_100 task")
            return
        self._shot_stop.set()
        print("[opengame] stopping screenshot_100...")
        if wait:
            try:
                self._shot_thread.join(timeout=3.0)
            except Exception:
                pass

    @staticmethod
    def _resolve_linux_ip(linux_ip: str) -> str:
        """解析 Linux 接收地址。"""
        if linux_ip != "auto":
            return linux_ip
        try:
            proc = subprocess.run(["hostname", "-I"], check=True, capture_output=True, text=True)
            parts = [item.strip() for item in proc.stdout.split() if item.strip()]
            if parts:
                return parts[0]
        except Exception:
            pass
        return "127.0.0.1"

    @staticmethod
    def _decode_ps_bytes(data: bytes) -> str:
        """解码 PowerShell 输出，兼容常见编码。"""
        if not data:
            return ""
        for enc in ("utf-8", "cp936", "gbk", "utf-16le"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _ps_invoke_cmd(args: list[str]) -> str:
        """将参数列表安全拼接为 PowerShell 外部命令调用脚本。"""
        if not args:
            return ""
        quoted = ["'" + str(a).replace("'", "''") + "'" for a in args]
        return "& " + " ".join(quoted)

    def _run_ps(self, script: str, check: bool = True) -> subprocess.CompletedProcess:
        """执行 PowerShell 脚本。"""
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=False,
        )
        out = self._decode_ps_bytes(proc.stdout)
        err = self._decode_ps_bytes(proc.stderr)
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output=out, stderr=err)
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)

    @staticmethod
    def _to_windows_path(path: str) -> str:
        """将 WSL 路径转换为 Windows 路径，便于 powershell/ffmpeg 写文件。"""
        try:
            p = subprocess.run(["wslpath", "-w", path], check=True, capture_output=True, text=True)
            out = (p.stdout or "").strip()
            if out:
                return out
        except Exception:
            pass
        return path

    def _list_window_titles(self) -> list[str]:
        """列出可见窗口标题。"""
        ps = r'''
Get-Process -ErrorAction SilentlyContinue |
Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -and $_.MainWindowTitle.Trim().Length -gt 0 } |
Select-Object -ExpandProperty MainWindowTitle -Unique |
Sort-Object |
ForEach-Object { Write-Output $_ }
'''
        try:
            proc = self._run_ps(ps, check=True)
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        except Exception:
            return []

    def _resolve_window_title(self) -> Optional[str]:
        """解析要抓取的窗口标题。"""
        titles = self._list_window_titles()
        if not titles:
            return None

        wanted = []
        if self.window_title and self.window_title.lower() != "auto":
            wanted.append(self.window_title)
        wanted.extend([
            "反恐精英",
            "全球攻势",
            "Counter-Strike",
            "Counter Strike",
            "CS2",
        ])

        lowered = [(title, title.lower()) for title in titles]
        for key in wanted:
            key_lower = key.lower()
            for title, title_lower in lowered:
                if key_lower in title_lower:
                    return title
        return None

    def _resolve_window_title_with_retry(self, timeout_sec: float = 20.0, interval_sec: float = 0.5) -> Optional[str]:
        """轮询窗口标题，避免游戏窗口出现较慢。"""
        deadline = time.time() + max(0.0, float(timeout_sec))
        while True:
            title = self._resolve_window_title()
            if title:
                return title
            if time.time() >= deadline:
                break
            time.sleep(max(0.1, float(interval_sec)))
        return None

    def _resolve_window_hwnd(self, preferred_title: Optional[str] = None) -> Optional[str]:
        """通过进程句柄定位游戏窗口 hwnd。"""
        title_match = ""
        if preferred_title:
            title_ps = preferred_title.replace("'", "''")
            title_match = (
                "$pt='" + title_ps + "'; "
                "$p = Get-Process -ErrorAction SilentlyContinue | "
                "Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -and $_.MainWindowTitle -like ('*' + $pt + '*') } | "
                "Select-Object -First 1; "
                "if ($p) { Write-Output ('0x{0:X}' -f $p.MainWindowHandle); exit 0 }; "
            )

        ps = (
            title_match
            + r'''
$p = Get-Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -and (
            $_.MainWindowTitle -match 'Counter-Strike|CS2|反恐精英|全球攻势'
        )
    } |
    Select-Object -First 1
if ($p) { Write-Output ('0x{0:X}' -f $p.MainWindowHandle) }
'''
        )
        try:
            proc = self._run_ps(ps, check=True)
            text = (proc.stdout or "").strip()
            if text:
                return text.splitlines()[-1].strip()
        except Exception:
            pass
        return None

    def _resolve_rect_from_hwnd(self, hwnd_hex: str) -> Optional[tuple[int, int, int, int]]:
        """根据窗口句柄读取窗口矩形，返回 (x,y,w,h)。"""
        if not hwnd_hex:
            return None
        hwnd_ps = hwnd_hex.replace("'", "''")
        ps = (
            "Add-Type -TypeDefinition @\"\n"
            "using System;\n"
            "using System.Runtime.InteropServices;\n"
            "public static class WRect {\n"
            "  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }\n"
            "  [DllImport(\"user32.dll\")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);\n"
            "}\n"
            "\"@ -Language CSharp; "
            "$h='" + hwnd_ps + "'; "
            "$v=[Int64]0; "
            "if($h -match '^0x'){ $v=[Convert]::ToInt64($h.Substring(2),16) } else { [void][Int64]::TryParse($h,[ref]$v) }; "
            "$r=New-Object WRect+RECT; "
            "$ptr=[IntPtr]$v; "
            "if([WRect]::GetWindowRect($ptr, [ref]$r)){ "
            "  $w=$r.Right-$r.Left; $hgt=$r.Bottom-$r.Top; "
            "  if($w -gt 0 -and $hgt -gt 0){ Write-Output (\"{0},{1},{2},{3}\" -f $r.Left,$r.Top,$w,$hgt) } "
            "}"
        )
        try:
            proc = self._run_ps(ps, check=True)
            text = (proc.stdout or "").strip()
            if not text:
                return None
            parts = text.splitlines()[-1].split(",")
            if len(parts) != 4:
                return None
            x, y, w, h = [int(p.strip()) for p in parts]
            if w <= 0 or h <= 0:
                return None
            return x, y, w, h
        except Exception:
            return None

    def _resolve_rect_from_title_winapi(self, title: str) -> Optional[tuple[int, int, int, int]]:
        """通过 Win32 枚举窗口，按标题匹配并读取矩形。"""
        if not title:
            return None
        title_ps = title.replace("'", "''")
        ps = (
            "Add-Type -TypeDefinition @\"\n"
            "using System;\n"
            "using System.Text;\n"
            "using System.Runtime.InteropServices;\n"
            "public static class WinSearch {\n"
            "  public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);\n"
            "  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }\n"
            "  [DllImport(\"user32.dll\")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);\n"
            "  [DllImport(\"user32.dll\")] public static extern bool IsWindowVisible(IntPtr hWnd);\n"
            "  [DllImport(\"user32.dll\", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int maxCount);\n"
            "  [DllImport(\"user32.dll\")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);\n"
            "}\n"
            "\"@ -Language CSharp; "
            "$kw='" + title_ps + "'; "
            "$kwL=$kw.ToLowerInvariant(); "
            "$found=$null; "
            "[WinSearch]::EnumWindows({ param($hWnd, $lParam) "
            "  if (-not [WinSearch]::IsWindowVisible($hWnd)) { return $true } "
            "  $sb = New-Object System.Text.StringBuilder 512; "
            "  [void][WinSearch]::GetWindowText($hWnd, $sb, $sb.Capacity); "
            "  $t = $sb.ToString().Trim(); "
            "  if ($t.Length -gt 0 -and $t.ToLowerInvariant().Contains($kwL)) { $script:found=$hWnd; return $false } "
            "  return $true "
            "}, [IntPtr]::Zero) | Out-Null; "
            "if ($found -ne $null) { "
            "  $r=New-Object WinSearch+RECT; "
            "  if([WinSearch]::GetWindowRect($found, [ref]$r)){ "
            "    $w=$r.Right-$r.Left; $h=$r.Bottom-$r.Top; "
            "    if($w -gt 0 -and $h -gt 0){ Write-Output (\"{0},{1},{2},{3}\" -f $r.Left,$r.Top,$w,$h) } "
            "  } "
            "}"
        )
        try:
            proc = self._run_ps(ps, check=True)
            text = (proc.stdout or "").strip()
            if not text:
                return None
            parts = text.splitlines()[-1].split(",")
            if len(parts) != 4:
                return None
            x, y, w, h = [int(p.strip()) for p in parts]
            if w <= 0 or h <= 0:
                return None
            return x, y, w, h
        except Exception:
            return None

    @staticmethod
    def _normalize_rect_for_encoder(rect: tuple[int, int, int, int]) -> Optional[tuple[int, int, int, int]]:
        """将窗口矩形规整为编码器可接受的偶数宽高。"""
        x, y, w, h = rect
        if w <= 1 or h <= 1:
            return None
        if w % 2 != 0:
            w -= 1
        if h % 2 != 0:
            h -= 1
        if w <= 1 or h <= 1:
            return None
        return x, y, w, h

    def _build_start_game_ps(self) -> str:
        """构造启动游戏 PowerShell 命令。"""
        exe = self.game_exe.replace("'", "''")
        if self.game_args:
            safe_args = ["'" + a.replace("'", "''") + "'" for a in self.game_args]
            return f"Start-Process -FilePath '{exe}' -ArgumentList {','.join(safe_args)}"
        return f"Start-Process -FilePath '{exe}'"

    def open_game(self, wait_seconds: float = 6.0) -> None:
        """打开 Windows 游戏进程。"""
        self._run_ps(self._build_start_game_ps(), check=True)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _build_ffmpeg_cmd(self, with_viewer: bool, window_rect: Optional[tuple[int, int, int, int]] = None) -> list[str]:
        """构造 ffmpeg 推流命令（窗口优先，使用窗口矩形裁剪 desktop）。"""
        outputs = list(self.stream_outputs)
        if with_viewer and self.viewer_source == self.win_viewer_src and self.win_viewer_src not in outputs:
            outputs.append(self.win_viewer_src)
        output_is_tee = len(outputs) > 1
        stream_output = outputs[0]
        if output_is_tee:
            stream_output = "|".join(f"[f=mpegts]{dest}" for dest in outputs)

        cmd = [
            self.ffmpeg_path,
            "-f",
            "gdigrab",
            "-framerate",
            str(self.framerate),
        ]

        if window_rect is not None:
            x, y, w, h = window_rect
            cmd += ["-offset_x", str(x), "-offset_y", str(y), "-video_size", f"{w}x{h}", "-i", "desktop"]
        else:
            cmd += ["-i", f"title={self.window_title}"]

        if output_is_tee:
            cmd += ["-map", "0:v:0"]

        cmd += [
            "-vcodec",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-x264-params",
            "repeat-headers=1:scenecut=0",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            self.bitrate,
            "-f",
            "tee" if output_is_tee else "mpegts",
            stream_output,
        ]
        return cmd

    def capture_screenshot(self, save_path: str) -> bool:
        """抓取一帧当前游戏窗口画面到本地文件。"""
        title = self.window_title if (self.window_title and self.window_title.lower() != "auto") else None
        if not title:
            title = self._resolve_window_title_with_retry(timeout_sec=10.0, interval_sec=0.4)
        if not title:
            print("[opengame] screenshot failed: game window title not found")
            return False

        self.window_title = title
        hwnd = self._resolve_window_hwnd(preferred_title=self.window_title)
        rect = self._resolve_rect_from_hwnd(hwnd) if hwnd else None
        if rect is None:
            rect = self._resolve_rect_from_title_winapi(self.window_title)
        if rect is not None:
            rect = self._normalize_rect_for_encoder(rect)

        out_abs = os.path.abspath(save_path)
        out_dir = os.path.dirname(out_abs)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_win = self._to_windows_path(out_abs)

        cmd = [self.ffmpeg_path, "-y", "-f", "gdigrab", "-framerate", "1"]
        if rect is not None:
            x, y, w, h = rect
            cmd += ["-offset_x", str(x), "-offset_y", str(y), "-video_size", f"{w}x{h}", "-i", "desktop"]
            print(f"[opengame] screenshot source -> window-rect={x},{y},{w},{h}")
        else:
            cmd += ["-i", f"title={self.window_title}"]
            print(f"[opengame] screenshot source -> title={self.window_title}")

        cmd += ["-frames:v", "1", "-q:v", "2", out_win]

        try:
            ps_cmd = self._ps_invoke_cmd(cmd)
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print("[opengame] screenshot ffmpeg failed")
                if proc.stderr:
                    print(proc.stderr.strip())
                return False
        except Exception as e:
            print(f"[opengame] screenshot failed: {e}")
            return False

        if os.path.exists(out_abs):
            print(f"[opengame] screenshot saved -> {out_abs}")
            return True
        print(f"[opengame] screenshot may be saved on Windows path -> {out_win}")
        return False

    @staticmethod
    def _tail_err(pipe, max_bytes: int = 4000) -> str:
        """读取并截断错误输出。"""
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

    def _spawn_ffmpeg_with_quick_check(self, cmd: list[str], probe_sec: float = 1.0) -> tuple[subprocess.Popen, Optional[str]]:
        """启动 ffmpeg 并做快速存活检查。"""
        ps_cmd = self._ps_invoke_cmd(cmd)
        p = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(max(0.2, float(probe_sec)))
        if p.poll() is None:
            return p, None
        return p, self._tail_err(p.stderr)

    def start_stream(self, with_viewer: bool) -> Optional[subprocess.Popen]:
        """启动实时推流。"""
        with self._lock:
            if self._stream_proc is not None:
                return self._stream_proc

            # 稳定性优化：若已有上次成功标题，优先复用，减少重启恢复时间。
            title = self.window_title if (self.window_title and self.window_title.lower() != "auto") else None
            if not title:
                title = self._resolve_window_title_with_retry(timeout_sec=25.0, interval_sec=0.5)
            if not title:
                print("[opengame] game window title not found; stream not started")
                print("[opengame] hint: pass --window-title with an exact/unique substring")
                print("[opengame] hint: if black screen, set game to borderless/windowed")
                return None

            self.window_title = title
            print(f"[opengame] resolved window title -> {self.window_title}")
            print("[opengame] capture backend -> gdigrab")
            hwnd = self._resolve_window_hwnd(preferred_title=self.window_title)
            rect = self._resolve_rect_from_hwnd(hwnd) if hwnd else None
            if rect is None:
                rect = self._resolve_rect_from_title_winapi(self.window_title)
            if rect is not None:
                norm = self._normalize_rect_for_encoder(rect)
                if norm is not None:
                    if norm != rect:
                        ox, oy, ow, oh = rect
                        nx, ny, nw, nh = norm
                        print(f"[opengame] normalize window-rect for encoder -> ({ox},{oy},{ow},{oh}) => ({nx},{ny},{nw},{nh})")
                    rect = norm
                else:
                    rect = None

            if rect is not None:
                x, y, w, h = rect
                print(f"[opengame] capture source -> window-rect={x},{y},{w},{h}")
            else:
                print(f"[opengame] capture source -> title={self.window_title}")

            cmd = self._build_ffmpeg_cmd(with_viewer=with_viewer, window_rect=rect)

            max_attempts = 3
            p = None
            err = None
            for attempt in range(1, max_attempts + 1):
                p_try, err_try = self._spawn_ffmpeg_with_quick_check(cmd, probe_sec=1.0)
                if err_try is None:
                    p = p_try
                    err = None
                    break
                p = p_try
                err = err_try
                if attempt < max_attempts:
                    print(f"[opengame] ffmpeg exited early on attempt {attempt}/{max_attempts}, retrying...")
                    time.sleep(0.8)

            if p is None or p.poll() is not None:
                print(f"[opengame] window capture failed (title={self.window_title})")
                print(f"[opengame] ffmpeg cmd -> {' '.join(cmd)}")
                if err:
                    print("[opengame] ffmpeg exited early:")
                    print(err.strip())
                self._stream_proc = None
                return None

            self._stream_proc = p
            print(f"[opengame] stream started -> {self.stream_dest}")
            return p

    def restart_stream(self, with_viewer: bool, cooldown_sec: float = 1.0) -> bool:
        """重启推流进程。"""
        self.stop_stream()
        if cooldown_sec > 0:
            time.sleep(cooldown_sec)
        p = self.start_stream(with_viewer=with_viewer)
        return p is not None

    def start_windows_viewer(self) -> bool:
        """在 Windows 侧启动 ffplay 预览。"""
        ffplay_exe = self.ffplay_path
        while "\\\\" in ffplay_exe:
            ffplay_exe = ffplay_exe.replace("\\\\", "\\")

        ffplay_exe_ps = ffplay_exe.replace("'", "''")
        win_args = [
            "-x",
            str(self.view_width),
            "-y",
            str(self.view_height),
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
            self.viewer_source,
        ]
        args_ps = ",".join("'" + a.replace("'", "''") + "'" for a in win_args)
        ps = (
            f"$exe='{ffplay_exe_ps}'; "
            "if (-not (Test-Path $exe)) { Write-Output '__ERR__NOEXE__'; exit 2 }; "
            f"$args=@({args_ps}); "
            "Start-Process -FilePath $exe -ArgumentList $args -WindowStyle Normal; "
            "Start-Sleep -Milliseconds 300; "
            "if (Get-Process -Name ffplay -ErrorAction SilentlyContinue) { Write-Output '__OK__' } else { Write-Output '__ERR__NOPROC__' }"
        )

        proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=False, capture_output=True, text=True)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        if "__OK__" in out:
            print("[opengame] viewer started (winffplay) on Windows desktop")
            return True
        if "__ERR__NOEXE__" in out:
            print(f"[opengame] winffplay failed: ffplay executable not found -> {self.ffplay_path}")
            return False
        if err:
            print(f"[opengame] winffplay stderr: {err}")
        print("[opengame] winffplay failed to start")
        return False

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

    def stop_windows_viewer(self) -> None:
        """关闭 Windows 侧 ffplay。"""
        try:
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "Get-Process -Name ffplay -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force }",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def run_forever(self, wait_game_seconds: float = 6.0, start_viewer: bool = True, open_game_first: bool = True) -> None:
        """一键启动并保持运行，直到 Ctrl+C。"""
        if open_game_first:
            self.open_game(wait_seconds=wait_game_seconds)
        stream = self.start_stream(with_viewer=start_viewer)
        if stream is None:
            print("[opengame] stream did not start, skip viewer startup")
            return

        if start_viewer:
            time.sleep(0.2)
            self.start_windows_viewer()

        print(f"Streaming to {self.stream_dest} (viewer={'on' if start_viewer else 'off'})")
        print("Press Ctrl+C to stop stream/viewer")
        print("Type 'screenshot' then Enter to capture one frame to ./screenshots/")
        print("Type 'screenshot_100' then Enter to capture 100 frames every 2 seconds")
        print("Type 'p' then Enter to stop screenshot_100")
        self._start_command_listener()
        restart_count = 0
        max_restarts = 20
        try:
            while True:
                try:
                    while True:
                        cmd = self._cmd_queue.get_nowait()
                        self._handle_runtime_command(cmd)
                except queue.Empty:
                    pass

                if self._stream_proc is not None and self._stream_proc.poll() is not None:
                    err = self._tail_err(self._stream_proc.stderr)
                    print("[opengame] stream process exited")
                    if err:
                        print("[opengame] stream stderr:")
                        print(err.strip())

                    if restart_count >= max_restarts:
                        print(f"[opengame] restart limit reached ({max_restarts}), stop streaming")
                        break

                    restart_count += 1
                    print(f"[opengame] restarting stream ({restart_count}/{max_restarts})...")
                    ok = self.restart_stream(with_viewer=start_viewer, cooldown_sec=1.0)
                    if not ok:
                        print("[opengame] stream restart failed, retry in 2s")
                        time.sleep(2.0)
                    continue
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_command_listener()
            self._stop_batch_screenshot(wait=True)
            self.stop_windows_viewer()
            self.stop_stream()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Open game on Windows and stream CS window to Linux")
    p.add_argument("--game-exe", required=False, help="Windows game executable path")
    p.add_argument("--game-arg", action="append", default=[], help="Game argument, repeatable")
    p.add_argument("--linux-ip", default="auto", help="Linux receiver IP used by Windows ffmpeg, default auto")
    p.add_argument("--port", type=int, default=12345)
    p.add_argument("--window-title", default="auto", help="Window title for gdigrab, default auto")
    p.add_argument("--stream-output", action="append", default=[], help="Raw stream output URL, repeatable")
    p.add_argument("--viewer-source", default="", help="Viewer input source URL (can be processed stream from YOLO/OCR)")
    p.add_argument("--framerate", type=int, default=60)
    p.add_argument("--bitrate", default="8M")
    p.add_argument("--view-width", type=int, default=800, help="Viewer window width")
    p.add_argument("--view-height", type=int, default=450, help="Viewer window height")
    p.add_argument("--wait-game", type=float, default=6.0, help="Seconds to wait after opening game")
    p.add_argument("--no-viewer", action="store_true", help="Do not start winffplay viewer")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffplay", default="ffplay")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if not args.game_exe:
        raise SystemExit("--game-exe is required")

    tool = OpenGameTool(
        game_exe=args.game_exe,
        game_args=args.game_arg,
        linux_ip=args.linux_ip,
        port=args.port,
        ffmpeg_path=args.ffmpeg,
        ffplay_path=args.ffplay,
        framerate=args.framerate,
        bitrate=args.bitrate,
        window_title=args.window_title,
        stream_outputs=args.stream_output,
        viewer_source=args.viewer_source or None,
        view_width=args.view_width,
        view_height=args.view_height,
    )
    tool.run_forever(
        wait_game_seconds=args.wait_game,
        start_viewer=not args.no_viewer,
        open_game_first=True,
    )


if __name__ == "__main__":
    main()
