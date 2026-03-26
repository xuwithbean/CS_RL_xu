"""自动打开 Windows 游戏并实时传输到 Linux。

功能：
- 在 WSL 中通过 `powershell.exe` 启动 Windows 游戏进程。
- 在 Windows 中通过 `ffmpeg + gdigrab` 采集窗口/桌面并 UDP 推流到 Linux。
- 可选在 Linux 侧启动 `ffplay` 查看，或在 Windows 侧启动 `ffplay` 弹窗查看。
- Ctrl+C 时自动停止推流与播放器进程（不强制关闭游戏进程）。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import threading
import time
from typing import Optional


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
        viewer_mode: str = "winffplay",
        framerate: int = 60,
        bitrate: str = "8M",
        capture_backend: str = "ddagrab",
        window_title: Optional[str] = None,
        use_desktop_capture: bool = False,
        allow_desktop_fallback: bool = False,
        monitor: int = 2,
        view_width: int = 800,
        view_height: int = 600,
    ):
        self.game_exe = game_exe
        self.game_args = game_args or []
        self.linux_ip = self._resolve_linux_ip(linux_ip)
        self.port = int(port)
        self.ffmpeg_path = ffmpeg_path
        self.ffplay_path = ffplay_path
        self.viewer_mode = viewer_mode
        self.framerate = int(framerate)
        self.bitrate = bitrate
        self.capture_backend = capture_backend
        self.window_title = window_title
        self.use_desktop_capture = bool(use_desktop_capture)
        self.allow_desktop_fallback = bool(allow_desktop_fallback)
        self.window_source: Optional[str] = None
        self.monitor = int(monitor)
        self.view_width = int(view_width)
        self.view_height = int(view_height)

        self.stream_dest = f"udp://{self.linux_ip}:{self.port}"
        self.viewer_src = f"udp://0.0.0.0:{self.port}"
        self.win_viewer_src = f"udp://127.0.0.1:{self.port}"

        self._stream_proc: Optional[subprocess.Popen] = None
        self._viewer_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _get_monitor_bounds(self, monitor_index: int) -> Optional[tuple[int, int, int, int]]:
        """查询 Windows 指定显示器边界，返回 (x, y, w, h)。"""
        if monitor_index <= 0:
            return None
        ps = r'''
Add-Type -AssemblyName System.Windows.Forms;
$screens = [System.Windows.Forms.Screen]::AllScreens
if ($screens.Count -lt 1) { exit 1 }
$idx = ''' + str(int(monitor_index - 1)) + r'''
if ($idx -lt 0 -or $idx -ge $screens.Count) { exit 2 }
$b = $screens[$idx].Bounds
Write-Output ("{0},{1},{2},{3}" -f $b.X, $b.Y, $b.Width, $b.Height)
'''
        try:
            proc = self._run_ps(ps, check=True)
            line = (proc.stdout or "").strip().splitlines()
            if not line:
                return None
            parts = line[-1].strip().split(",")
            if len(parts) != 4:
                return None
            x, y, w, h = [int(p) for p in parts]
            return x, y, w, h
        except Exception:
            return None

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
            titles = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if titles:
                return titles
        except Exception:
            pass

        # 兼容路径：某些环境下 EnumWindows 可能失败，改为进程主窗口标题枚举。
        ps2 = r'''
$titles = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -and $_.MainWindowTitle.Trim().Length -gt 0 } |
    Select-Object -ExpandProperty MainWindowTitle -Unique
$titles | Sort-Object -Unique | ForEach-Object { Write-Output $_ }
'''
        try:
            proc2 = self._run_ps(ps2, check=True)
            titles2 = [line.strip() for line in proc2.stdout.splitlines() if line.strip()]
            if titles2:
                return titles2
        except Exception:
            pass

        # 再次兜底：直接用 gdigrab 列窗口，避免 PowerShell 枚举在某些会话下拿不到标题。
        return self._list_windows_from_gdigrab()

    def _list_windows_from_gdigrab(self) -> list[str]:
        """通过 Windows ffmpeg(gdigrab -list_windows) 获取窗口标题列表。"""
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-f",
            "gdigrab",
            "-list_windows",
            "true",
            "-i",
            "desktop",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        safe_cmd = " ".join(subprocess.list2cmdline([c]) for c in cmd)
        ps_cmd = f"& {{ {safe_cmd} }}"
        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
                check=False,
                capture_output=True,
                text=True,
            )
            text = "\n".join([(proc.stdout or ""), (proc.stderr or "")])
        except Exception:
            return []

        titles: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if "gdigrab" in low and ("error" in low or "warning" in low):
                continue

            title = ""
            if "\"" in line:
                l = line.find("\"")
                r = line.rfind("\"")
                if r > l:
                    title = line[l + 1:r].strip()
            elif line.startswith("0x"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    title = parts[1].strip()
            elif "title=" in low:
                idx = low.find("title=")
                title = line[idx + len("title="):].strip()

            title = title.strip().strip("'").strip("\"")
            if not title:
                continue
            if title.lower() in ("desktop", "program manager"):
                continue
            titles.append(title)

        uniq: list[str] = []
        seen = set()
        for t in titles:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(t)
        return uniq

    def _guess_process_name_candidates(self) -> list[str]:
        """推测游戏进程名候选（不含 .exe）。"""
        names: list[str] = []
        try:
            base = os.path.basename(self.game_exe or "").strip()
            if base.lower().endswith(".exe"):
                base = base[:-4]
            if base:
                names.append(base)
        except Exception:
            pass
        names.extend(["cs2", "csgo", "Counter-Strike"])

        uniq: list[str] = []
        seen = set()
        for n in names:
            k = n.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(n)
        return uniq

    def _guess_window_title_candidates(self) -> list[str]:
        """在无法枚举窗口时，基于已知信息盲探测标题。"""
        names = []
        names.extend(self._list_windows_from_gdigrab())
        if self.window_title and self.window_title.lower() != "auto":
            names.append(self.window_title)
        names.extend([
            "Counter-Strike 2",
            "Counter-Strike",
            "Counter Strike 2",
            "CS2",
            "cs2",
            "反恐精英2",
            "反恐精英",
        ])

        uniq = []
        seen = set()
        for n in names:
            key = n.strip().lower()
            if key and key not in seen:
                seen.add(key)
                uniq.append(n.strip())
        return uniq

    def _resolve_window_hwnd(self) -> Optional[str]:
        """尝试通过进程 MainWindowHandle 定位游戏窗口句柄，返回形如 0x1A2B3C。"""
        candidates = self._guess_process_name_candidates()
        cand_ps = ",".join("'" + c.replace("'", "''") + "'" for c in candidates)
        ps = (
            "$cands=@(" + cand_ps + "); "
            "$procs=Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 }; "
            "$hit=$null; "
            "foreach($c in $cands){ $hit=$procs | Where-Object { $_.ProcessName -ieq $c } | Select-Object -First 1; if($hit){ break } }; "
            "if(-not $hit){ $hit=$procs | Where-Object { $_.MainWindowTitle -match 'Counter-Strike|CS2|反恐精英|全球攻势' } | Select-Object -First 1 }; "
            "if($hit){ Write-Output ('0x{0:X}' -f $hit.MainWindowHandle) }"
        )
        try:
            proc = self._run_ps(ps, check=True)
            text = (proc.stdout or "").strip()
            if text:
                return text.splitlines()[-1].strip()
        except Exception:
            pass
        return None

    def _resolve_title_from_hwnd(self, hwnd_hex: str) -> Optional[str]:
        """根据窗口句柄读取窗口标题。"""
        if not hwnd_hex:
            return None
        hwnd_ps = hwnd_hex.replace("'", "''")
        ps = (
            "$h='" + hwnd_ps + "'; "
            "$p=Get-Process -ErrorAction SilentlyContinue | "
            "Where-Object { ('0x{0:X}' -f $_.MainWindowHandle) -ieq $h } | Select-Object -First 1; "
            "if($p -and $p.MainWindowTitle -and $p.MainWindowTitle.Trim().Length -gt 0){ Write-Output $p.MainWindowTitle }"
        )
        try:
            proc = self._run_ps(ps, check=True)
            title = (proc.stdout or "").strip()
            return title if title else None
        except Exception:
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

    def _resolve_rect_from_title_probe(self, title: str) -> Optional[tuple[int, int, int, int]]:
        """通过 gdigrab title 探测日志反推出窗口矩形。"""
        if not title:
            return None
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-f",
            "gdigrab",
            "-framerate",
            "1",
            "-i",
            f"title={title}",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        safe_cmd = " ".join(subprocess.list2cmdline([c]) for c in cmd)
        ps_cmd = f"& {{ {safe_cmd} }}"
        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", ps_cmd],
                check=False,
                capture_output=True,
                text=True,
            )
            text = "\n".join([(proc.stdout or ""), (proc.stderr or "")])
        except Exception:
            return None

        # 例："capturing 1920x1080x32 at (0,0)"
        m = re.search(r"capturing\s+(\d+)x(\d+)x\d+\s+at\s+\((-?\d+),\s*(-?\d+)\)", text, re.IGNORECASE)
        if not m:
            return None
        w = int(m.group(1))
        h = int(m.group(2))
        x = int(m.group(3))
        y = int(m.group(4))
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h

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

    def _resolve_window_title_with_retry(self, timeout_sec: float = 12.0, interval_sec: float = 0.5) -> Optional[str]:
        """在启动初期轮询窗口标题，减少游戏窗口出现较慢导致的漏检。"""
        deadline = time.time() + max(0.0, float(timeout_sec))
        while True:
            title = self._resolve_window_title()
            if title:
                return title
            if time.time() >= deadline:
                break
            time.sleep(max(0.1, float(interval_sec)))
        return None

    def _resolve_window_hwnd_with_retry(self, timeout_sec: float = 8.0, interval_sec: float = 0.5) -> Optional[str]:
        """轮询窗口句柄，适配标题不可见但窗口句柄可见的场景。"""
        deadline = time.time() + max(0.0, float(timeout_sec))
        while True:
            hwnd = self._resolve_window_hwnd()
            if hwnd:
                return hwnd
            if time.time() >= deadline:
                break
            time.sleep(max(0.1, float(interval_sec)))
        return None

    def _probe_window_title_stream(self, candidate_title: str) -> Optional[subprocess.Popen]:
        """尝试用单个标题启动推流，成功返回进程。"""
        self.window_title = candidate_title
        self.window_source = f"title={candidate_title}"
        try:
            cmd_probe = self._build_ffmpeg_cmd()
        except RuntimeError:
            return None

        safe_probe = " ".join(subprocess.list2cmdline([c]) for c in cmd_probe)
        ps_probe = f"& {{ {safe_probe} }}"
        p_probe = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", ps_probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(1.0)
        if p_probe.poll() is None:
            return p_probe

        err_probe = self._tail_err(p_probe.stderr)
        if err_probe and "Error opening input" not in err_probe:
            print(f"[opengame] title probe failed ({candidate_title}):")
            print(err_probe.strip())
        return None

    def _resolve_game_window_source(self, title_retry_sec: float = 15.0, hwnd_retry_sec: float = 8.0, allow_blind_probe: bool = True) -> Optional[subprocess.Popen]:
        """解析游戏窗口采集源。

        返回值：
        - `subprocess.Popen`：已通过盲探测直接启动了推流进程。
        - `None`：仅完成窗口源解析（或失败，需调用方检查 `self.window_source`）。
        """
        resolved_title = self._resolve_window_title_with_retry(timeout_sec=title_retry_sec, interval_sec=0.6)
        if resolved_title:
            if resolved_title != self.window_title:
                print(f"[opengame] resolved window title -> {resolved_title}")
            self.window_title = resolved_title
            self.window_source = f"title={resolved_title}"
            # 对游戏窗口优先改用 rect 裁剪 desktop，规避 title 采集黑屏。
            rect = self._resolve_rect_from_title_probe(resolved_title)
            if rect is None:
                resolved_hwnd = self._resolve_window_hwnd_with_retry(timeout_sec=3.0, interval_sec=0.3)
                if resolved_hwnd:
                    rect = self._resolve_rect_from_hwnd(resolved_hwnd)
            if rect:
                x, y, w, h = rect
                self.window_source = f"rect={x},{y},{w},{h}"
                print(f"[opengame] switched capture source to rect -> x={x}, y={y}, w={w}, h={h}")
        elif self.window_title and self.window_title.lower() != "auto":
            print(f"[opengame] requested title not found -> {self.window_title}")
            self.window_source = None
        else:
            self.window_title = None
            self.window_source = None

        if self.window_source is None:
            resolved_hwnd = self._resolve_window_hwnd_with_retry(timeout_sec=hwnd_retry_sec, interval_sec=0.5)
            if resolved_hwnd:
                # 对硬件加速游戏窗口，title 抓取常见黑屏；优先使用 rect 裁剪 desktop。
                rect = self._resolve_rect_from_hwnd(resolved_hwnd)
                if rect:
                    x, y, w, h = rect
                    self.window_source = f"rect={x},{y},{w},{h}"
                    print(f"[opengame] resolved game window rect -> x={x}, y={y}, w={w}, h={h}")
                else:
                    resolved_title2 = self._resolve_title_from_hwnd(resolved_hwnd)
                    if resolved_title2:
                        self.window_title = resolved_title2
                        self.window_source = f"title={resolved_title2}"
                        print(f"[opengame] resolved game window by hwnd -> title={resolved_title2}")

        if self.window_source is not None:
            return None

        if self.allow_desktop_fallback:
            print("[opengame] game window title not found; fallback to desktop capture is enabled")
            self.use_desktop_capture = True
            return None

        if allow_blind_probe:
            candidates = self._guess_window_title_candidates()
            print("[opengame] window enumeration failed, trying blind title probes...")
            for cand in candidates:
                p_probe = self._probe_window_title_stream(cand)
                if p_probe is not None:
                    print(f"[opengame] stream started by title probe -> {cand}")
                    return p_probe

        self.window_source = None
        print("[opengame] game window title not found; stream not started")
        print("[opengame] hint: pass --window-title with an exact/unique substring, or use --allow-desktop-fallback")
        print("[opengame] hint: game may be running in exclusive fullscreen; switch to borderless/windowed mode")
        self._print_window_title_candidates()
        return None

    def _print_window_title_candidates(self, max_count: int = 20) -> None:
        """打印可见窗口标题候选，便于用户设置 --window-title。"""
        titles = self._list_window_titles()
        if not titles:
            print("[opengame] visible window titles: <none>")
            return
        print("[opengame] visible window title candidates:")
        for idx, title in enumerate(titles[:max_count], start=1):
            print(f"  {idx}. {title}")
        if len(titles) > max_count:
            print(f"  ... ({len(titles) - max_count} more)")

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

    @staticmethod
    def _decode_ps_bytes(data: bytes) -> str:
        """解码 PowerShell 输出，兼容 UTF-8 / GBK 等本地编码。"""
        if not data:
            return ""
        for enc in ("utf-8", "cp936", "gbk", "utf-16le"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")

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
        output_is_tee = self.viewer_mode == "winffplay"
        stream_output = self.stream_dest
        if output_is_tee:
            stream_output = f"[f=mpegts]{self.stream_dest}|[f=mpegts]{self.win_viewer_src}"

        if self.capture_backend == "ddagrab":
            cmd = [
                self.ffmpeg_path,
                "-f",
                "ddagrab",
                "-framerate",
                str(self.framerate),
                "-i",
                "output_idx=0",
            ]
            vf = None
            if self.use_desktop_capture:
                bounds = self._get_monitor_bounds(self.monitor)
                if bounds is not None:
                    x, y, w, h = bounds
                    vf = f"crop={w}:{h}:{x}:{y}"
            else:
                source = self.window_source
                if source and source.startswith("rect="):
                    parts = source[len("rect="):].split(",")
                    if len(parts) != 4:
                        raise RuntimeError("invalid rect window source")
                    x, y, w, h = [int(p.strip()) for p in parts]
                    vf = f"crop={w}:{h}:{x}:{y}"

            if vf:
                cmd += ["-vf", vf]

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
                "tee" if output_is_tee else "mpegts",
                stream_output,
            ]
            return cmd

        cmd = [
            self.ffmpeg_path,
            "-f",
            "gdigrab",
            "-framerate",
            str(self.framerate),
        ]

        if self.use_desktop_capture:
            bounds = self._get_monitor_bounds(self.monitor)
            if bounds is not None:
                x, y, w, h = bounds
                cmd += ["-offset_x", str(x), "-offset_y", str(y), "-video_size", f"{w}x{h}"]
            cmd += ["-i", "desktop"]
        else:
            source = self.window_source
            if source and source.startswith("rect="):
                parts = source[len("rect="):].split(",")
                if len(parts) != 4:
                    raise RuntimeError("invalid rect window source")
                x, y, w, h = [int(p.strip()) for p in parts]
                cmd += ["-offset_x", str(x), "-offset_y", str(y), "-video_size", f"{w}x{h}", "-i", "desktop"]
            elif source:
                cmd += ["-i", source]
            elif self.window_title and self.window_title.lower() != "auto":
                cmd += ["-i", f"title={self.window_title}"]
            else:
                raise RuntimeError("window source is not resolved; game-window capture cannot start")

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
            "tee" if output_is_tee else "mpegts",
            stream_output,
        ]
        return cmd

    def _build_ffplay_cmd(self) -> list[str]:
        """构造 ffplay 低延迟查看命令。"""
        return [
            self.ffplay_path,
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
            self.viewer_src,
        ]

    def _build_windows_ffplay_cmd(self) -> list[str]:
        """构造 Windows 侧 ffplay 弹窗命令。

        使用 PowerShell Start-Process 启动，播放本机 UDP 推流而非再次抓屏。
        """
        ffplay_exe = self.ffplay_path
        # 容错：run.sh 里常传入双反斜杠路径，这里归一化成单反斜杠 Windows 路径。
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
            self.win_viewer_src,
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
        return ["powershell.exe", "-NoProfile", "-Command", ps]

    def _is_windows_ffplay_running(self) -> bool:
        """检查 Windows 侧是否存在 ffplay 进程。"""
        try:
            proc = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "if (Get-Process -Name ffplay -ErrorAction SilentlyContinue) { Write-Output 1 } else { Write-Output 0 }",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip().startswith("1")
        except Exception:
            return False

    def start_stream(self, restart_if_running: bool = False) -> Optional[subprocess.Popen]:
        """启动实时推流（Windows -> Linux UDP）。"""
        with self._lock:
            if self._stream_proc is not None:
                if not restart_if_running:
                    return self._stream_proc
                self.stop_stream()

            if not self.use_desktop_capture:
                probed = self._resolve_game_window_source(title_retry_sec=45.0, hwnd_retry_sec=20.0, allow_blind_probe=True)
                if probed is not None:
                    self._stream_proc = probed
                    return probed
                if self.window_source is None and not self.use_desktop_capture:
                    self._stream_proc = None
                    return None

            try:
                cmd = self._build_ffmpeg_cmd()
            except RuntimeError as e:
                print(f"[opengame] {e}")
                self._stream_proc = None
                return None
            print(f"[opengame] capture backend -> {self.capture_backend}")
            if self.use_desktop_capture:
                print("[opengame] capture source -> desktop")
            elif self.window_source:
                print(f"[opengame] capture source -> {self.window_source}")
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
                if self.capture_backend == "ddagrab" and "Unknown input format: 'ddagrab'" in (err or ""):
                    print("[opengame] ddagrab is not supported by current ffmpeg, fallback to gdigrab")
                    self.capture_backend = "gdigrab"
                    try:
                        cmd_retry = self._build_ffmpeg_cmd()
                    except RuntimeError as e:
                        print(f"[opengame] {e}")
                        self._stream_proc = None
                        return None
                    safe_retry = " ".join(subprocess.list2cmdline([c]) for c in cmd_retry)
                    ps_retry = f"& {{ {safe_retry} }}"
                    p_retry = subprocess.Popen(
                        ["powershell.exe", "-NoProfile", "-Command", ps_retry],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    time.sleep(1.0)
                    if p_retry.poll() is None:
                        self._stream_proc = p_retry
                        print(f"[opengame] stream started -> {self.stream_dest}")
                        return p_retry
                    err_retry = self._tail_err(p_retry.stderr)
                    if err_retry:
                        print("[opengame] ffmpeg retry error:")
                        print(err_retry.strip())
                    self._stream_proc = None
                    return None

                # 尝试自动回退：窗口标题采集失败时切到 desktop 采集
                if not self.use_desktop_capture and self.allow_desktop_fallback:
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
                    self._stream_proc = None
                    return None

                if not self.use_desktop_capture and not self.allow_desktop_fallback:
                    print(f"[opengame] window capture failed (title={self.window_title})")
                    print("[opengame] desktop fallback is disabled")

                if err:
                    print("[opengame] ffmpeg exited early:")
                    print(err.strip())
                self._stream_proc = None
                return None
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

            if self.viewer_mode == "winffplay":
                try:
                    if not self.use_desktop_capture and self.window_source is None:
                        title = self._resolve_window_title_with_retry(timeout_sec=5.0, interval_sec=0.5)
                        if title:
                            self.window_title = title
                            self.window_source = f"title={title}"
                        else:
                            hwnd = self._resolve_window_hwnd_with_retry(timeout_sec=4.0, interval_sec=0.5)
                            if hwnd:
                                title2 = self._resolve_title_from_hwnd(hwnd)
                                if title2:
                                    self.window_title = title2
                                    self.window_source = f"title={title2}"
                                else:
                                    rect = self._resolve_rect_from_hwnd(hwnd)
                                    if rect:
                                        x, y, w, h = rect
                                        self.window_source = f"rect={x},{y},{w},{h}"
                    cmd = self._build_windows_ffplay_cmd()
                except RuntimeError as e:
                    print(f"[opengame] {e}")
                    self._print_window_title_candidates()
                    self._viewer_proc = None
                    return None
            else:
                cmd = self._build_ffplay_cmd()

            if self.viewer_mode == "winffplay":
                # `start` 会立刻返回，实际窗口在 Windows 独立进程中。
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                out = (proc.stdout or "").strip()
                err = (proc.stderr or "").strip()

                if "__OK__" in out:
                    self._viewer_proc = None
                    print("[opengame] viewer started (winffplay) on Windows desktop")
                    return None

                if "__ERR__NOEXE__" in out:
                    print(f"[opengame] winffplay failed: ffplay executable not found -> {self.ffplay_path}")
                    if err:
                        print("[opengame] powershell error:")
                        print(err)
                    self._viewer_proc = None
                    return None

                if "__ERR__NOPROC__" in out:
                    print("[opengame] winffplay failed: process not found after Start-Process")
                    if err:
                        print("[opengame] powershell error:")
                        print(err)
                    self._viewer_proc = None
                    return None

                time.sleep(0.8)
                if self._is_windows_ffplay_running():
                    self._viewer_proc = None
                    print("[opengame] viewer started (winffplay) on Windows desktop")
                    return None
                print("[opengame] winffplay may have failed to start.")
                if out:
                    print(f"[opengame] powershell stdout: {out}")
                if err:
                    print(f"[opengame] powershell stderr: {err}")
                self._viewer_proc = None
                return None

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
            else:
                print(f"[opengame] viewer started ({self.viewer_mode}) <- {self.viewer_src}")
            self._viewer_proc = p
            return p

    def stop_viewer(self) -> None:
        """停止 Linux 侧查看器。"""
        if self.viewer_mode == "winffplay":
            # Windows 侧弹窗预览，按进程名关闭。
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
            return

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

    def start_all(self, wait_game_seconds: float = 6.0, start_viewer: bool = True) -> bool:
        """先启动游戏，再启动推流，并可选启动 Linux 查看器。"""
        self.open_game(wait_seconds=wait_game_seconds)
        stream_proc = self.start_stream(restart_if_running=True)
        if stream_proc is None:
            print("[opengame] stream did not start, skip viewer startup")
            return False
        if start_viewer:
            # 给推流一个很短的准备时间
            time.sleep(0.2)
            self.start_viewer(restart_if_running=True)
        return True

    def stop_all(self) -> None:
        """停止查看器与推流（不强制结束游戏进程）。"""
        self.stop_viewer()
        self.stop_stream()

    def run_forever(self, wait_game_seconds: float = 6.0, start_viewer: bool = True) -> None:
        """一键启动并保持运行，直到 Ctrl+C。"""
        started = self.start_all(wait_game_seconds=wait_game_seconds, start_viewer=start_viewer)
        if not started:
            return
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
    p.add_argument("--game-exe", required=False, help="Windows game executable path")
    p.add_argument("--game-arg", action="append", default=[], help="Game argument, repeatable")
    p.add_argument("--linux-ip", default="auto", help="Linux receiver IP used by Windows ffmpeg, default auto")
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--window-title", default="auto", help="Window title for gdigrab, default auto")
    p.add_argument("--desktop", action="store_true", help="Capture desktop instead of window title")
    p.add_argument("--allow-desktop-fallback", action="store_true", help="Allow fallback to desktop capture when window capture fails")
    p.add_argument("--framerate", type=int, default=60)
    p.add_argument("--bitrate", default="8M")
    p.add_argument("--capture-backend", choices=["gdigrab", "ddagrab"], default="ddagrab", help="Capture backend for ffmpeg")
    p.add_argument("--monitor", type=int, default=2, help="Desktop capture monitor index (1-based)")
    p.add_argument("--view-width", type=int, default=800, help="Viewer window width")
    p.add_argument("--view-height", type=int, default=600, help="Viewer window height")
    p.add_argument("--wait-game", type=float, default=6.0, help="Seconds to wait after opening game")
    p.add_argument("--no-viewer", action="store_true", help="Do not start viewer")
    p.add_argument("--viewer-mode", choices=["ffplay", "winffplay"], default="winffplay")
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
        viewer_mode=args.viewer_mode,
        framerate=args.framerate,
        bitrate=args.bitrate,
        capture_backend=args.capture_backend,
        monitor=args.monitor,
        view_width=args.view_width,
        view_height=args.view_height,
        window_title=args.window_title,
        use_desktop_capture=args.desktop,
        allow_desktop_fallback=args.allow_desktop_fallback,
    )
    tool.run_forever(wait_game_seconds=args.wait_game, start_viewer=not args.no_viewer)


if __name__ == "__main__":
    main()
