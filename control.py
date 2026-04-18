# [x]: 完成游戏内操作控制（低延迟统一实现）
"""WSL -> Windows 低延迟控制模块。

设计目标：
- 统一键盘与鼠标控制实现，避免重复代码。
- 仅保留 socket 控制通道（优先低延迟），不再走 PowerShell 回退。
- 复用长连接并支持批量命令，减少每次操作的连接开销。
"""

from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Iterable, Literal, Optional


def _detect_windows_host() -> str:
    """在 WSL 中探测 Windows 主机地址。"""
    # 优先使用默认路由网关（通常是 Windows 主机侧 vEthernet 地址）。
    try:
        p = subprocess.run(
            ["ip", "route", "show", "default"],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in (p.stdout or "").splitlines():
            parts = line.strip().split()
            if "via" in parts:
                idx = parts.index("via")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
    except Exception:
        pass

    # 常见场景：/etc/resolv.conf 的 nameserver 即 Windows 主机侧地址。
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("nameserver "):
                    parts = s.split()
                    if len(parts) >= 2:
                        return parts[1]
    except Exception:
        pass
    return "127.0.0.1"


def _env_host() -> str:
    return os.getenv("CONTROL_HOST", _detect_windows_host())


def _env_port() -> int:
    try:
        # 50999 在部分 Windows 环境会命中保留端口策略，默认改为更常用的高位端口。
        return int(os.getenv("CONTROL_PORT", "60000"))
    except Exception:
        return 60000


def _env_auto_start() -> bool:
    return os.getenv("CONTROL_AUTO_START", "1").strip().lower() not in {"0", "false", "no", "off"}


def _env_wait_ready() -> float:
    try:
        return float(os.getenv("CONTROL_WAIT_READY", "25"))
    except Exception:
        return 25.0


class WinControlClient:
    """Windows 控制服务器客户端（行协议，长连接）。"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 0.5,
    ):
        self.host = host or _env_host()
        self.port = int(port or _env_port())
        self.timeout = float(timeout)
        self.auto_start_listener = _env_auto_start()
        self._sock: Optional[socket.socket] = None
        self._rw = None
        self._lock = threading.Lock()
        self._autostart_attempted = False

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        """关闭连接（调用方需已持有 self._lock）。"""
        if self._rw is not None:
            try:
                self._rw.close()
            except Exception:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._rw = None
        self._sock = None

    def _connect_locked(self) -> None:
        if self._sock is not None and self._rw is not None:
            return
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self._sock = s
        self._rw = s.makefile("rwb")

    def send_lines(self, lines: Iterable[str], expect_reply: bool = False) -> Optional[str]:
        """发送多行命令并读取服务端回复。"""
        cmd_lines = [ln for ln in lines if ln]
        if not cmd_lines:
            return None

        with self._lock:
            for attempt in range(2):
                try:
                    self._connect_locked()
                    for ln in cmd_lines:
                        self._rw.write((ln + "\n").encode("utf-8"))
                    self._rw.flush()

                    last = None
                    for _ in cmd_lines:
                        last = self._rw.readline().decode("utf-8", errors="ignore").strip()
                    return last if expect_reply else None
                except Exception as e:
                    self._close_locked()
                    if attempt == 0 and self.auto_start_listener and not self._autostart_attempted:
                        self._autostart_attempted = True
                        print("[control] listener not ready, auto starting...")
                        try:
                            self.port = start_server(port=self.port)
                            continue
                        except Exception as start_err:
                            print(f"[control] auto start listener failed: {start_err}")
                    if isinstance(e, ConnectionRefusedError):
                        print("[control] connection refused: Windows control listener is not running.")
                        print("[control] start it via: python control.py")
                    return None
        return None

    def send(self, line: str, expect_reply: bool = False) -> Optional[str]:
        return self.send_lines([line], expect_reply=expect_reply)


_SHARED_CLIENT: Optional[WinControlClient] = None


def get_shared_client() -> WinControlClient:
    """返回模块级共享客户端，降低多对象间的连接与握手开销。"""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = WinControlClient()
    return _SHARED_CLIENT


VK_MAP = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    " ": 0x20,
    "space": 0x20,
    "ctrl": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "tab": 0x09,
    "esc": 0x1B,
}


def _normalize_keys(key: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(key, (list, tuple)):
        return [str(k).strip() for k in key if str(k).strip()]
    if isinstance(key, str):
        if "+" in key:
            return [part.strip() for part in key.split("+") if part.strip()]
        if key.strip():
            return [key.strip()]
    raise TypeError("key must be str or list/tuple of str")


def _char_to_vk(ch: str) -> int:
    ch = ch.lower()
    if ch in VK_MAP:
        return VK_MAP[ch]
    if len(ch) == 1 and "a" <= ch <= "z":
        return ord(ch.upper())
    if len(ch) == 1 and "0" <= ch <= "9":
        return ord(ch)
    raise ValueError(f"Unsupported key: {ch}")


class KeySender:
    """低延迟键盘控制（基于 socket 命令批处理）。"""

    def __init__(self, default_hold_ms: int = 50, client: Optional[WinControlClient] = None):
        self.default_hold_ms = int(default_hold_ms)
        self.client = client or get_shared_client()

    def press(self, key: str | list[str] | tuple[str, ...]) -> None:
        self.press_and_release(key)

    def release(self, key: str | list[str] | tuple[str, ...]) -> None:
        vks = [_char_to_vk(k) for k in _normalize_keys(key)]
        self.client.send_lines([f"KEY_UP {vk}" for vk in reversed(vks)])

    def press_and_release(
        self,
        key: str | list[str] | tuple[str, ...],
        hold_ms: int | None = None,
        inter_ms: int = 0,
    ) -> None:
        if hold_ms is None:
            hold_ms = self.default_hold_ms
        vks = [_char_to_vk(k) for k in _normalize_keys(key)]
        lines = []
        if inter_ms and int(inter_ms) > 0:
            lines.append(f"SLEEP {int(inter_ms)}")
        lines.extend(f"KEY_DOWN {vk}" for vk in vks)
        lines.append(f"SLEEP {int(hold_ms)}")
        lines.extend(f"KEY_UP {vk}" for vk in reversed(vks))
        self.client.send_lines(lines)


class MouseController:
    """低延迟鼠标控制（基于 socket 命令批处理）。"""

    def __init__(self, default_hold_ms: int = 50, client: Optional[WinControlClient] = None):
        self.default_hold_ms = int(default_hold_ms)
        self.client = client or get_shared_client()

    def move(self, dx: int, dy: int) -> None:
        self.client.send(f"MOUSE_MOVE {int(dx)} {int(dy)}")

    def click(
        self,
        button: Literal["left", "right", "middle"] = "left",
        hold_ms: int | None = None,
        inter_ms: int = 0,
    ) -> None:
        if hold_ms is None:
            hold_ms = self.default_hold_ms
        # 支持点击前短暂停顿，便于与移动等动作衔接。
        if inter_ms and int(inter_ms) > 0:
            self.client.send_lines([
                f"SLEEP {int(inter_ms)}",
                f"MOUSE_CLICK {button} {int(hold_ms)}",
            ])
            return
        self.client.send(f"MOUSE_CLICK {button} {int(hold_ms)}")

    def move_and_click(
        self,
        dx: int,
        dy: int,
        button: Literal["left", "right", "middle"] = "left",
        hold_ms: int | None = None,
        inter_ms: int = 5,
    ) -> None:
        if hold_ms is None:
            hold_ms = self.default_hold_ms
        lines = [
            f"MOUSE_MOVE {int(dx)} {int(dy)}",
            f"SLEEP {int(inter_ms)}",
            f"MOUSE_PRESS {button}",
            f"SLEEP {int(hold_ms)}",
            f"MOUSE_RELEASE {button}",
        ]
        self.client.send_lines(lines)

    def press(self, button: Literal["left", "right", "middle"] = "left") -> None:
        self.client.send(f"MOUSE_PRESS {button}")

    def release(self, button: Literal["left", "right", "middle"] = "left") -> None:
        self.client.send(f"MOUSE_RELEASE {button}")

    def scroll(self, delta: int) -> None:
        self.client.send(f"MOUSE_SCROLL {int(delta)}")

    def is_button_pressed(self, button: Literal["x1", "x2", "left", "right", "middle"] = "x2") -> bool:
        vk_map = {"x1": 0x05, "x2": 0x06, "left": 0x01, "right": 0x02, "middle": 0x04}
        vk = vk_map[button.lower()]
        resp = self.client.send(f"IS_BUTTON {vk}", expect_reply=True)
        return bool(resp and resp.startswith("1"))


# 兼容旧接口
_default_client = get_shared_client()
_DEFAULT_SENDER = KeySender(client=_default_client)
_DEFAULT_MOUSE = MouseController(client=_default_client)


def send_key_windows(
    key: str | list[str] | tuple[str, ...],
    hold_ms: int = 50,
    inter_ms: int = 0,
) -> None:
    _DEFAULT_SENDER.press_and_release(key, hold_ms, inter_ms)


def mouse_move(dx: int, dy: int) -> None:
    _DEFAULT_MOUSE.move(dx, dy)


def mouse_click(
    button: Literal["left", "right", "middle"] = "left",
    hold_ms: int | None = None,
    inter_ms: int = 0,
) -> None:
    _DEFAULT_MOUSE.click(button, hold_ms, inter_ms)


def mouse_press(button: Literal["left", "right", "middle"] = "left") -> None:
    _DEFAULT_MOUSE.press(button)


def mouse_release(button: Literal["left", "right", "middle"] = "left") -> None:
    _DEFAULT_MOUSE.release(button)


def mouse_scroll(delta: int) -> None:
    _DEFAULT_MOUSE.scroll(delta)


PS_SERVER = r'''
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
namespace KE {
    public static class WinAPI {
        [DllImport("user32.dll", SetLastError=true)]
        public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
        [DllImport("user32.dll", SetLastError=true)]
        public static extern void mouse_event(uint dwFlags, uint dx, uint dy, int dwData, UIntPtr dwExtraInfo);
        [DllImport("user32.dll")]
        public static extern short GetAsyncKeyState(int vKey);
    }
}
"@ -Language CSharp

$candidatePorts = @({ports})
$listener = $null
$lastBindError = ''
foreach ($candidatePort in $candidatePorts) {
    foreach ($bindAddr in @([System.Net.IPAddress]::Any, [System.Net.IPAddress]::Loopback)) {
        try {
            $listener = [System.Net.Sockets.TcpListener]::new($bindAddr, [int]$candidatePort)
            $listener.Start()
            break
        } catch {
            $lastBindError = $_.Exception.Message
            try { if ($listener -ne $null) { $listener.Stop() } } catch {}
            $listener = $null
        }
    }
    if ($listener -ne $null) {
        break
    }
}
if ($listener -eq $null) {
    throw "Unable to bind control listener on any candidate port: $($candidatePorts -join ','); lastError=$lastBindError"
}
try {
    while ($true) {
        $client = $listener.AcceptTcpClient()
        $ns = $client.GetStream()
        $sr = New-Object System.IO.StreamReader($ns)
        $sw = New-Object System.IO.StreamWriter($ns)
        $sw.AutoFlush = $true
        try {
            while (($line = $sr.ReadLine()) -ne $null) {
                $cmd = $line.Trim()
                if ($cmd -eq '') { continue }
                $parts = $cmd.Split(' ')
                $verb = $parts[0].ToUpperInvariant()
                switch ($verb) {
                    'PING' { $sw.WriteLine('PONG') }
                    'SHUTDOWN' { $sw.WriteLine('OK'); $client.Close(); $listener.Stop(); exit 0 }
                    'SLEEP' { Start-Sleep -Milliseconds ([int]$parts[1]); $sw.WriteLine('OK') }
                    'KEY_DOWN' { [KE.WinAPI]::keybd_event([byte][int]$parts[1],0,0,[UIntPtr]::Zero); $sw.WriteLine('OK') }
                    'KEY_UP' { [KE.WinAPI]::keybd_event([byte][int]$parts[1],0,2,[UIntPtr]::Zero); $sw.WriteLine('OK') }
                    'MOUSE_MOVE' { [KE.WinAPI]::mouse_event(0x0001, [uint32][int]$parts[1], [uint32][int]$parts[2], 0, [UIntPtr]::Zero); $sw.WriteLine('OK') }
                    'MOUSE_PRESS' {
                        switch ($parts[1].ToLower()) {
                            'left' { [KE.WinAPI]::mouse_event(0x0002,0,0,0,[UIntPtr]::Zero) }
                            'right' { [KE.WinAPI]::mouse_event(0x0008,0,0,0,[UIntPtr]::Zero) }
                            'middle' { [KE.WinAPI]::mouse_event(0x0020,0,0,0,[UIntPtr]::Zero) }
                        }
                        $sw.WriteLine('OK')
                    }
                    'MOUSE_RELEASE' {
                        switch ($parts[1].ToLower()) {
                            'left' { [KE.WinAPI]::mouse_event(0x0004,0,0,0,[UIntPtr]::Zero) }
                            'right' { [KE.WinAPI]::mouse_event(0x0010,0,0,0,[UIntPtr]::Zero) }
                            'middle' { [KE.WinAPI]::mouse_event(0x0040,0,0,0,[UIntPtr]::Zero) }
                        }
                        $sw.WriteLine('OK')
                    }
                    'MOUSE_CLICK' {
                        $button = $parts[1].ToLower()
                        $hold = if ($parts.Length -gt 2) { [int]$parts[2] } else { 50 }
                        switch ($button) {
                            'left' { [KE.WinAPI]::mouse_event(0x0002,0,0,0,[UIntPtr]::Zero); Start-Sleep -Milliseconds $hold; [KE.WinAPI]::mouse_event(0x0004,0,0,0,[UIntPtr]::Zero) }
                            'right' { [KE.WinAPI]::mouse_event(0x0008,0,0,0,[UIntPtr]::Zero); Start-Sleep -Milliseconds $hold; [KE.WinAPI]::mouse_event(0x0010,0,0,0,[UIntPtr]::Zero) }
                            'middle' { [KE.WinAPI]::mouse_event(0x0020,0,0,0,[UIntPtr]::Zero); Start-Sleep -Milliseconds $hold; [KE.WinAPI]::mouse_event(0x0040,0,0,0,[UIntPtr]::Zero) }
                        }
                        $sw.WriteLine('OK')
                    }
                    'MOUSE_SCROLL' { [KE.WinAPI]::mouse_event(0x0800,0,0,[int]$parts[1],[UIntPtr]::Zero); $sw.WriteLine('OK') }
                    'IS_BUTTON' { $s=[KE.WinAPI]::GetAsyncKeyState([int]$parts[1]); if (($s -band 0x8000) -ne 0) { $sw.WriteLine('1') } else { $sw.WriteLine('0') } }
                    Default { $sw.WriteLine('ERR Unknown command: '+$cmd) }
                }
            }
        } catch {
        } finally {
            try { $client.Close() } catch {}
        }
    }
} finally {
    try { $listener.Stop() } catch {}
}
'''


class StartWinListener:
    """在 WSL 中启动/停止 Windows 控制监听器。"""

    def __init__(self, port: Optional[int] = None, wait_ready: Optional[float] = None):
        self.port = int(port or _env_port())
        self.wait_ready = float(_env_wait_ready() if wait_ready is None else wait_ready)
        self.host = _env_host()

    def _candidate_ports(self) -> list[int]:
        ports = [self.port]
        for offset in range(1, 16):
            ports.append(self.port + offset)
        for offset in range(1, 6):
            if self.port - offset > 0:
                ports.append(self.port - offset)
        # 增加离散备选端口，规避系统保留端口区间。
        ports.extend([60000, 60100, 55000, 52000, 50000, 45000, 38080, 27015, 23456, 18080])
        return list(dict.fromkeys(p for p in ports if 0 < p < 65536))

    def _probe_hosts(self) -> list[str]:
        hosts = [self.host, "127.0.0.1", "localhost"]
        uniq = []
        seen = set()
        for h in hosts:
            if h and h not in seen:
                seen.add(h)
                uniq.append(h)
        return uniq

    @staticmethod
    def _encode_powershell(script: str) -> str:
        return base64.b64encode(script.encode("utf-16le")).decode("ascii")

    @staticmethod
    def _is_control_server_ready(host: str, port: int, timeout: float = 0.2) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout) as s:
                s.settimeout(timeout)
                s.sendall(b"PING\n")
                data = s.recv(64)
                if not data:
                    return False
                return data.decode("utf-8", errors="ignore").strip().upper().startswith("PONG")
        except Exception:
            return False

    def start(self) -> int:
        candidate_ports = self._candidate_ports()
        script = PS_SERVER.replace("{ports}", ",".join(str(p) for p in candidate_ports))
        b64 = self._encode_powershell(script)
        # 直接拉起监听脚本；若启动失败可拿到 stderr 便于定位。
        proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                b64,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        deadline = time.time() + self.wait_ready
        while time.time() < deadline:
            code = proc.poll()
            if code is not None:
                out_b, err_b = proc.communicate(timeout=0.2)

                def _decode_bytes(data: bytes | None) -> str:
                    if not data:
                        return ""
                    try:
                        return data.decode("utf-8").strip()
                    except Exception:
                        try:
                            return data.decode("gbk", errors="ignore").strip()
                        except Exception:
                            return data.decode(errors="ignore").strip()

                msg = (_decode_bytes(err_b) or _decode_bytes(out_b) or "").strip()
                if msg:
                    raise RuntimeError(f"Windows listener process exited early (code={code}): {msg}")
                raise RuntimeError(f"Windows listener process exited early (code={code})")
            for h in self._probe_hosts():
                for p in candidate_ports:
                    if self._is_control_server_ready(h, p, timeout=0.2):
                        self.host = h
                        self.port = p
                        return self.port
            time.sleep(0.05)
        raise RuntimeError(
            f"Windows server not listening on hosts={self._probe_hosts()} ports={candidate_ports} after {self.wait_ready}s"
        )

    def stop(self, timeout: float = 1.0) -> None:
        last_err = None
        for h in self._probe_hosts():
            try:
                with socket.create_connection((h, self.port), timeout=timeout) as s:
                    s.sendall(b"SHUTDOWN\n")
                    try:
                        s.recv(1024)
                    except Exception:
                        pass
                    return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to contact server on port {self.port}: {last_err}")


def start_server(port: Optional[int] = None, wait_ready: Optional[float] = None) -> int:
    return StartWinListener(port=port, wait_ready=wait_ready).start()


def stop_server(port: Optional[int] = None, timeout: float = 1.0) -> None:
    StartWinListener(port=port).stop(timeout=timeout)


def _main_listener_cli(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", "-p", type=int, default=_env_port())
    p.add_argument("--wait-ready", type=float, default=_env_wait_ready())
    p.add_argument("--stop", action="store_true")
    args = p.parse_args(argv)
    if args.stop:
        stop_server(port=args.port)
        print("Sent shutdown")
        return 0
    print("Starting Windows listener...")
    listener = StartWinListener(port=args.port, wait_ready=args.wait_ready)
    actual_port = listener.start()
    print(f"Windows listener started on {listener.host}:{actual_port}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main_listener_cli())
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        raise SystemExit(1)
