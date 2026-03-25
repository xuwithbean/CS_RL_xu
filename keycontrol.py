# [x]: 完成对Windows的键盘控制操作。

"""WSL -> Windows 键发送工具（最小化版本）。

此文件仅保留 `VK_MAP` 和模块级发送函数 `send_key_windows`。
"""

import subprocess
import socket
import time

SERVER_HOST = '192.168.221.36'
SERVER_PORT = 54321


def _send_to_server(lines, expect_reply=False, timeout=0.2):
    """Send one or many lines to the Windows control server. `lines` may be str or list[str].
    Returns response text if expect_reply=True and server replies; otherwise None.
    """
    if isinstance(lines, str):
        lines = [lines]
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=timeout) as s:
            f = s.makefile('rwb')
            for ln in lines:
                f.write((ln + '\n').encode('utf-8'))
            f.flush()
            if expect_reply:
                # read one line reply
                resp = f.readline().decode('utf-8', errors='ignore')
                return resp.strip()
            return None
    except Exception:
        return None


class KeySender:
    """提供按下/抬起分离的键发送器。"""

    def __init__(
        self, 
        default_hold_ms: int = 50,
        VK_MAP = {
            'w': 0x57,
            'a': 0x41,
            's': 0x53,
            'd': 0x44,
            ' ': 0x20,
            'ctrl': 0x11,
            'shift': 0x10,
            'alt': 0x12,
            'tab': 0x09,
            'esc': 0x1B,
        },
    ):
        self.default_hold_ms = int(default_hold_ms)
        self.VK_MAP=VK_MAP
    def _char_to_vk(self,ch: str) -> int:
        ch = ch.lower()
        if ch in self.VK_MAP:
            return self.VK_MAP[ch]
        if len(ch) == 1 and 'a' <= ch <= 'z':
            return ord(ch.upper())
        if len(ch) == 1 and '0' <= ch <= '9':
            return ord(ch)
        raise ValueError(f"Unsupported key: {ch}")


    def _build_powershell_header(self) -> str:
        return (
            'Add-Type -TypeDefinition @"\n'
            'using System;\n'
            'using System.Runtime.InteropServices;\n'
            'namespace KE {\n'
            '    public class K {\n'
            '        [DllImport("user32.dll", SetLastError=true)]\n'
            '        public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);\n'
            '    }\n'
            '}\n'
            '"@ -Language CSharp;\n'
        )
    def _normalize_keys(self, key: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(key, (list, tuple)):
            keys = [str(k) for k in key]
        elif isinstance(key, str):
            if '+' in key:
                keys = [part.strip() for part in key.split('+') if part.strip()]
            else:
                keys = [key]
        else:
            raise TypeError('key must be str or list/tuple of str')
        return keys

    def press(self, key: str | list[str] | tuple[str, ...]) -> None:
        """按下一个或多个键（只发 keydown）。

        参数与之前 `send_key_windows` 的 `key` 格式一致。
        """
        keys = self._normalize_keys(key)
        try:
            vks = [self._char_to_vk(k) for k in keys]
        except ValueError as e:
            print(e)
            return

        # try server first (faster); fall back to direct PowerShell if server not available
        lines = [f'KEY_DOWN {vk}' for vk in vks]
        if _send_to_server(lines) is not None:
            return

        header = self._build_powershell_header()
        down_lines = '\n'.join(f'[KE.K]::keybd_event([byte]{vk},0,0,[UIntPtr]::Zero);' for vk in vks) + '\n'
        ps = header + down_lines
        try:
            subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to press keys via PowerShell: {e}")

    def release(self, key: str | list[str] | tuple[str, ...]) -> None:
        """释放一个或多个键（只发 keyup），释放顺序为给定顺序的反向。"""
        keys = self._normalize_keys(key)
        try:
            vks = [self._char_to_vk(k) for k in keys]
        except ValueError as e:
            print(e)
            return

        lines = [f'KEY_UP {vk}' for vk in reversed(vks)]
        if _send_to_server(lines) is not None:
            return

        header = self._build_powershell_header()
        up_lines = '\n'.join(f'[KE.K]::keybd_event([byte]{vk},0,2,[UIntPtr]::Zero);' for vk in reversed(vks)) + '\n'
        ps = header + up_lines
        try:
            subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to release keys via PowerShell: {e}")

    def press_and_release(self, key: str | list[str] | tuple[str, ...], hold_ms: int | None = None) -> None:
        """按下（按顺序）、等待 hold_ms 毫秒、然后释放（反向顺序）。"""
        if hold_ms is None:
            hold_ms = self.default_hold_ms
        keys = self._normalize_keys(key)
        try:
            vks = [self._char_to_vk(k) for k in keys]
        except ValueError as e:
            print(e)
            return

        # send sequence to server: KEY_DOWN... SLEEP hold_ms KEY_UP...
        lines = []
        for vk in vks:
            lines.append(f'KEY_DOWN {vk}')
        lines.append(f'SLEEP {int(hold_ms)}')
        for vk in reversed(vks):
            lines.append(f'KEY_UP {vk}')
        if _send_to_server(lines) is not None:
            return

        header = self._build_powershell_header()
        down = '\n'.join(f'[KE.K]::keybd_event([byte]{vk},0,0,[UIntPtr]::Zero);' for vk in vks)
        up = '\n'.join(f'[KE.K]::keybd_event([byte]{vk},0,2,[UIntPtr]::Zero);' for vk in reversed(vks))
        body = down + '\n' + f'Start-Sleep -Milliseconds {int(hold_ms)};' + '\n' + up + '\n'
        ps = header + body
        try:
            subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to press_and_release via PowerShell: {e}")


# 模块级向后兼容封装
_DEFAULT_SENDER = KeySender()


def send_key_windows(key: str | list[str] | tuple[str, ...], hold_ms: int = 50) -> None:
    """向后兼容的单次按下-等待-释放接口。"""
    _DEFAULT_SENDER.press_and_release(key, hold_ms)
