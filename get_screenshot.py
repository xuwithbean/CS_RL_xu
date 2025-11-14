"""
低延迟屏幕流（Windows -> WSL）
本模块重构为基于实时流的方案：使用 Windows 上的 ffmpeg 抓取桌面并通过 UDP 推送，
在 WSL 上使用 ffplay/ffmpeg 接收并播放。使用多线程管理后台进程与状态，避免磁盘 I/O。
主要接口：
- ScreenshotTool.start_stream(...)
- ScreenshotTool.stop_stream()
- ScreenshotTool.start_viewer(...)
- ScreenshotTool.stop_viewer()
- ScreenshotTool.start_end_to_end(...)  # 同时启动流和查看
注意：
- 要求 Windows 上可执行 `ffmpeg.exe`（放到 PATH 或在 `ffmpeg_path` 指定完整路径）。
- WSL 可用 `ffplay` 或 `ffmpeg` 来接收并播放 udp://192.168.221.36:PORT。
此实现去掉了基于磁盘的图片读写路径，优先低延迟网络流方法。
"""
import subprocess
import threading
import time
import traceback
from typing import Optional
import os
class ScreenshotTool:
    """低延迟屏幕流工具（Windows + WSL）。
    设计要点：
    - 使用多线程管理后台进程（streamer 和 viewer）。
    - 不再通过文件轮询读取图片，而是通过 UDP 流传输 H.264 视频以降低延迟。
    - 允许在出错时重启流或查看器。
    """
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffplay_path: str = "ffplay",
        port: int = 1234,
        framerate: int = 60,
        bitrate: str = "20k",
        ffmpeg_dest: Optional[str] = None,
        ffplay_source: Optional[str] = None,
        # 低延迟相关参数（可调）
        gop: Optional[int] = None,
        maxrate: Optional[str] = None,
        bufsize: Optional[str] = None,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.ffplay_path = ffplay_path
        self.port = int(port)
        self.framerate = int(framerate)
        self.bitrate = str(bitrate)
        # 可配置的默认目标/源 URL
        self.ffmpeg_dest = ffmpeg_dest or f"udp://192.168.221.36:{self.port}"
        self.ffplay_source = ffplay_source or f"udp://192.168.221.36:{self.port}"
        # 低延迟参数
        # GOP 长度（关键帧间隔），默认使用帧率（约 1 秒一关键帧），可以设置更小以降低解码延迟
        self.gop = int(gop) if gop is not None else int(self.framerate)
        # 最大码率与缓冲区（用于 x264 VBV/CBR 风格控制），可为空
        self.maxrate = maxrate
        self.bufsize = bufsize
        # 进程句柄与锁
        self._stream_proc: Optional[subprocess.Popen] = None
        self._viewer_proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        # 管理线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
    # -------------------- 构造命令 --------------------
    def _build_ffmpeg_cmd(self) -> list:
        """
        返回用于捕获桌面的 ffmpeg 命令参数列表。
        可选参数：若传入 monitor_bounds (x,y,w,h)，会将 capture 限定到指定显示器区域，
        使用 `-offset_x/-offset_y` 与 `-video_size` 实现只捕获某个显示器（例如桌面2）。
        """
        # 默认目标使用实例的 ffmpeg_dest
        dest = self.ffmpeg_dest
        base_cmd = [
            self.ffmpeg_path,
            "-f",
            "gdigrab",
            "-framerate",
            str(self.framerate),
        ]
        # Note: monitor-specific options (video_size/offset) will be inserted by caller when needed
        # 基本编码参数（优先低延迟）
        base_cmd += [
            "-i",
            "desktop",
            "-vcodec",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
        ]

        # 强制更短 GOP、关闭场景切换以降低延迟抖动
        if self.gop:
            base_cmd += ["-g", str(self.gop), "-keyint_min", str(max(1, self.gop // 2)), "-sc_threshold", "0"]

        # 设置码率/缓冲（可选）
        if self.bitrate:
            base_cmd += ["-b:v", self.bitrate]
        if self.maxrate:
            base_cmd += ["-maxrate", str(self.maxrate)]
        if self.bufsize:
            base_cmd += ["-bufsize", str(self.bufsize)]

        # x264 参数：强制尽量低延迟
        x264_params = []
        # 禁用重复帧积累，降低延迟的 x264 参数
        x264_params.append("no-scenecut=1")
        # 如果有 maxrate/bufsize，可设置 vbv
        if self.maxrate or self.bufsize:
            # 用 x264-params 来传递更多控制参数（保守组合）
            pass
        if x264_params:
            base_cmd += ["-x264-params", ":".join(x264_params)]

        base_cmd += ["-pix_fmt", "yuv420p", "-f", "mpegts", dest]
        return base_cmd
    def _get_monitor_bounds(self, monitor_index: int) -> Optional[tuple]:
        """
        通过 PowerShell 查询 Windows 上所有显示器的 Bounds，并返回指定 monitor_index (1-based) 的 (x,y,w,h)。
        返回 None 表示查询失败。此函数在 WSL 下通过调用 `powershell.exe` 获取信息。
        """
        try:
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "[System.Windows.Forms.Screen]::AllScreens | ForEach-Object { \"$($_.Bounds.X),$($_.Bounds.Y),$($_.Bounds.Width),$($_.Bounds.Height)\" }"
            )
            proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True, check=True)
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if not lines:
                return None
            idx = monitor_index - 1
            if idx < 0 or idx >= len(lines):
                return None
            parts = lines[idx].split(",")
            if len(parts) != 4:
                return None
            x, y, w, h = map(int, parts)
            return (x, y, w, h)
        except Exception:
            return None
    def start_stream_monitor(self, monitor: int = 2, dest: Optional[str] = None, background: bool = True) -> Optional[subprocess.Popen]:
        """
        启动 ffmpeg 并仅捕获指定 monitor（1-based 索引）的画面。
        - monitor: 要捕获的显示器索引（1-based），默认 2（桌面2）。
        - dest: 目标地址，例如 "udp://192.168.221.36:1234"；默认使用实例的 ffmpeg_dest。
        - background: True 时以子进程启动并返回 Popen 对象；False 时等待 ffmpeg 退出并返回 None。
        """
        target = dest or self.ffmpeg_dest
        bounds = self._get_monitor_bounds(monitor)
        if bounds is None:
            print(f"无法获取 monitor {monitor} 的边界信息，启动失败。")
            return None
        ox, oy, w, h = bounds
        cmd = [
            self.ffmpeg_path,
            "-f",
            "gdigrab",
            "-framerate",
            str(self.framerate),
            "-offset_x",
            str(ox),
            "-offset_y",
            str(oy),
            "-video_size",
            f"{w}x{h}",
            "-i",
            "desktop",
            "-vcodec",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mpegts",
            target,
        ]
        if self.bitrate:
            cmd += ["-b:v", self.bitrate]
        safe_cmd = " ".join(subprocess.list2cmdline([c]) for c in cmd)
        ps_cmd = f"& {{ {safe_cmd} }}"
        try:
            if background:
                p = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # 保存到实例以便后续 stop_end_to_end / stop_stream 能终止该进程
                with self._proc_lock:
                    self._stream_proc = p
                return p
            else:
                subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_cmd], check=False)
                return None
        except Exception:
            traceback.print_exc()
            return None
    def _build_ffplay_cmd(self) -> list:
        """返回 ffplay 的接收播放命令参数列表（在 WSL 环境）。使用实例的 ffplay_source 作为输入。"""
        src = self.ffplay_source
        # 减小探测与分析时间，降低启动延迟
        cmd = [
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
            src,
            "-infbuf",
        ]
        return cmd
    # -------------------- 启停流 --------------------
    def start_stream(self, restart_if_running: bool = False) -> Optional[subprocess.Popen]:
        """在 Windows 上启动 ffmpeg，将桌面实时推送到 udp://192.168.221.36:port。
        - 如果已经在运行，默认不重复启动；传入 restart_if_running=True 会先停止再启动。
        - 返回启动的 subprocess.Popen 对象，或 None。
        """
        with self._proc_lock:
            if self._stream_proc is not None:
                if not restart_if_running:
                    return self._stream_proc
                else:
                    self.stop_stream()
            try:
                cmd = self._build_ffmpeg_cmd()
                # 在 powershell 中执行命令串
                safe_cmd = " ".join(subprocess.list2cmdline([c]) for c in cmd)
                ps_cmd = f"& {{ {safe_cmd} }}"
                p = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._stream_proc = p
                return p
            except Exception as e:
                print("start_stream 启动失败:", e)
                traceback.print_exc()
                return None
    def stop_stream(self) -> None:
        """停止当前正在运行的流（如果存在）。"""
        with self._proc_lock:
            p = self._stream_proc
            self._stream_proc = None
        if p is None:
            return
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=1)
        except Exception:
            pass
        # 如果进程仍然存在（Windows 上可能产生子进程），尝试使用 PowerShell/ taskkill 强制结束
        try:
            if p.poll() is None:
                # 首先尝试使用 PowerShell Stop-Process
                try:
                    subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"Stop-Process -Id {p.pid} -Force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
                # 作为后备，使用 taskkill 强行杀死进程树（Windows 命令）
                try:
                    subprocess.run(["cmd.exe", "/C", f"taskkill /PID {p.pid} /T /F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
        except Exception:
            pass
        # 最后再尝试按进程名强制清理所有 ffmpeg 进程（保底措施）
        try:
            # PowerShell 尝试停止名为 ffmpeg 的进程
            try:
                subprocess.run(["powershell.exe", "-NoProfile", "-Command", "Get-Process -Name ffmpeg -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force }"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            # 备选：使用 taskkill 按进程名强制杀掉 ffmpeg.exe
            try:
                subprocess.run(["cmd.exe", "/C", "taskkill /IM ffmpeg.exe /F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        except Exception:
            pass
    # -------------------- 启停查看（viewer） --------------------
    def start_viewer(self, restart_if_running: bool = False, source: Optional[str] = None, width: int = 1280, height: int = 770) -> Optional[subprocess.Popen]:
        """在 WSL（Linux）侧启动 ffplay 来接收并播放 UDP 流。

        参数:
          - source: 覆盖默认的 ffplay 输入地址（例如 'udp://IP:PORT'）。
          - width,height: 初始窗口大小（像素），默认 1280x770。
        """
        with self._proc_lock:
            if self._viewer_proc is not None:
                if not restart_if_running:
                    return self._viewer_proc
                else:
                    self.stop_viewer()
            try:
                src = source or self.ffplay_source
                cmd = [
                    self.ffplay_path,
                    "-fflags",
                    "nobuffer",
                    "-flags",
                    "low_delay",
                    "-framedrop",
                    "-strict",
                    "-1",
                    "-x",
                    str(width),
                    "-y",
                    str(height),
                    src,
                    "-infbuf",
                ]
                p = subprocess.Popen(cmd)
                self._viewer_proc = p
                return p
            except Exception as e:
                print("start_viewer 启动失败:", e)
                traceback.print_exc()
                return None
    def stop_viewer(self) -> None:
        with self._proc_lock:
            p = self._viewer_proc
            self._viewer_proc = None
        if p is None:
            return
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=1)
        except Exception:
            pass
    # -------------------- 一键启动/停止（端到端） --------------------
    def start_end_to_end(self, with_viewer: bool = True) -> None:
        """同时在 Windows 启动流并（可选）在 WSL 启动 viewer。后台管理线程保证两者在意外退出时尝试重启。"""
        self.start_stream(restart_if_running=True)
        if with_viewer:
            self.start_viewer(restart_if_running=True)

        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    def stop_end_to_end(self) -> None:
        """停止监控线程并终止 viewer/stream。"""
        self._monitor_stop.set()
        self.stop_viewer()
        self.stop_stream()
        if self._monitor_thread is not None:
            try:
                self._monitor_thread.join(timeout=1)
            except Exception:
                pass
            self._monitor_thread = None
    def _monitor_loop(self) -> None:
        """后台监控：如果 stream/viewer 非期望运行则尝试重启。"""
        try:
            while not self._monitor_stop.is_set():
                with self._proc_lock:
                    sp = self._stream_proc
                    vp = self._viewer_proc
                if sp is not None and sp.poll() is not None:
                    try:
                        self._stream_proc = None
                    except Exception:
                        pass
                    try:
                        self.start_stream(restart_if_running=True)
                    except Exception:
                        pass
                if vp is not None and vp.poll() is not None:
                    try:
                        self._viewer_proc = None
                    except Exception:
                        pass
                    try:
                        self.start_viewer(restart_if_running=True)
                    except Exception:
                        pass
                time.sleep(0.5)
        except Exception:
            traceback.print_exc()
# 使用示例：
# st = ScreenshotTool()
# st.start_end_to_end()
# 结束时 st.stop_end_to_end()

    def get_one_frame(self, monitor: int = 2, return_pil: bool = True, timeout: int = 5):
        """捕获指定显示器的一帧并返回图像。

        - monitor: 1-based 显示器索引（与其他方法一致），默认 2。
        - return_pil: 若 True，尝试返回一个 PIL.Image 对象（需要 pillow 可用）；若 False 或 pillow 不可用，返回原始 PNG bytes。
        - timeout: 等待 ffmpeg 完成捕获的超时时间（秒）。

        实现：在 Windows 上用 ffmpeg/gdigrab 捕获一帧到 stdout（PNG），由 powershell 在 WSL 中调用。
        如果无法获取 monitor 边界，会尝试全桌面捕获。
        """
        bounds = self._get_monitor_bounds(monitor)
        cmd = [
            self.ffmpeg_path,
            "-f",
            "gdigrab",
            "-framerate",
            str(self.framerate),
        ]
        if bounds:
            ox, oy, w, h = bounds
            cmd += ["-offset_x", str(ox), "-offset_y", str(oy), "-video_size", f"{w}x{h}"]
        # single frame to stdout as PNG
        cmd += [
            "-i",
            "desktop",
            "-frames:v",
            "1",
            "-f",
            "image2",
            "-vcodec",
            "png",
            "pipe:1",
        ]

        safe_cmd = " ".join(subprocess.list2cmdline([c]) for c in cmd)
        ps_cmd = f"& {{ {safe_cmd} }}"
        try:
            # run powershell and capture stdout (binary)
            proc = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_cmd], capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                # 非零退出码，返回 None
                return None
            data = proc.stdout
            if not data:
                return None
            if return_pil:
                try:
                    from PIL import Image
                    import io

                    img = Image.open(io.BytesIO(data))
                    img.load()
                    return img
                except Exception:
                    # pillow 不可用或解码失败，返回原始 bytes
                    return data
            else:
                return data
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            traceback.print_exc()
            return None
