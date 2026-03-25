# [x]: 完成对Windows的鼠标控制操作。
"""WSL -> Windows 鼠标控制工具。

提供在 WSL 中调用的最小化鼠标控制类。通过 `powershell.exe` 使用 Add-Type 编译小段 C#，
调用 user32.dll::mouse_event 来发送鼠标事件。API 简洁：移动、按下、释放、点击、滚轮。

注意：需在目标 Windows 上保持窗口/桌面接受鼠标事件（焦点与权限相关）。
"""

import subprocess
import time
import socket
from typing import Literal

SERVER_HOST = '192.168.221.36'
SERVER_PORT = 54321


def _send_to_server(lines, expect_reply=False, timeout=0.2):
    if isinstance(lines, str):
        lines = [lines]
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=timeout) as s:
            f = s.makefile('rwb')
            for ln in lines:
                f.write((ln + '\n').encode('utf-8'))
            f.flush()
            if expect_reply:
                return f.readline().decode('utf-8', errors='ignore').strip()
            return None
    except Exception:
        return None


class MouseController:
    """简单鼠标控制封装（相对移动、按键、滚轮）。"""

    # mouse_event flags
    M_MOVE = 0x0001
    M_LDOWN = 0x0002
    M_LUP = 0x0004
    M_RDOWN = 0x0008
    M_RUP = 0x0010
    M_MDOWN = 0x0020
    M_MUP = 0x0040
    M_WHEEL = 0x0800
    M_ABSOLUTE = 0x8000

    def __init__(self, default_hold_ms: int = 50):
        self.default_hold_ms = int(default_hold_ms)

    def _build_powershell_header(self) -> str:
        # 与 control.py 保持一致的 Add-Type 源（注意这里使用单大括号）
        return (
            'Add-Type -TypeDefinition @"\n'
            'using System;\n'
            'using System.Runtime.InteropServices;\n'
            'namespace KE {\n'
            '    public class M {\n'
            '        [DllImport("user32.dll", SetLastError=true)]\n'
            '        public static extern void mouse_event(uint dwFlags, uint dx, uint dy, int dwData, UIntPtr dwExtraInfo);\n'
            '    }\n'
            '}\n'
            '"@ -Language CSharp;\n'
        )

    def _run_ps(self, body: str) -> None:
        ps = self._build_powershell_header() + body + '\n'
        try:
            subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to run PowerShell mouse command: {e}")

    def move(self, dx: int, dy: int) -> None:
        """相对移动鼠标（像素）。"""
        # try server first
        if _send_to_server(f'MOUSE_MOVE {int(dx)} {int(dy)}') is not None:
            return
        body = f'[KE.M]::mouse_event({self.M_MOVE}, {int(dx)}, {int(dy)}, 0, [UIntPtr]::Zero);'
        self._run_ps(body)

    def click(self, button: Literal['left', 'right', 'middle'] = 'left', hold_ms: int | None = None) -> None:
        """单次点击（按下-等待-释放）。"""
        if hold_ms is None:
            hold_ms = self.default_hold_ms
        if button == 'left':
            down, up = self.M_LDOWN, self.M_LUP
        elif button == 'right':
            down, up = self.M_RDOWN, self.M_RUP
        elif button == 'middle':
            down, up = self.M_MDOWN, self.M_MUP
        else:
            raise ValueError('button must be left/right/middle')

        # try server first
        if _send_to_server(f'MOUSE_CLICK {button} {int(hold_ms)}') is not None:
            return
        body = f'[KE.M]::mouse_event({down}, 0, 0, 0, [UIntPtr]::Zero);'
        body += '\n'
        body += f'Start-Sleep -Milliseconds {int(hold_ms)};\n'
        body += f'[KE.M]::mouse_event({up}, 0, 0, 0, [UIntPtr]::Zero);'
        self._run_ps(body)

    def move_and_click(self, dx: int, dy: int, button: Literal['left','right','middle']='left', hold_ms: int | None = None, inter_ms: int = 5) -> None:
        """在一次 PowerShell 调用中完成相对移动并点击（减少进程/编译开销）。

        - `inter_ms`：移动完毕与按下之间的短暂停顿（毫秒），默认 5ms。
        - `hold_ms`：按住时长（毫秒），默认使用构造器的 `default_hold_ms`。
        """
        if hold_ms is None:
            hold_ms = self.default_hold_ms

        if button == 'left':
            down, up = self.M_LDOWN, self.M_LUP
        elif button == 'right':
            down, up = self.M_RDOWN, self.M_RUP
        elif button == 'middle':
            down, up = self.M_MDOWN, self.M_MUP
        else:
            raise ValueError('button must be left/right/middle')

        # 构建单次 powershell body：移动 -> 按下 -> 等待 hold -> 释放
        parts = []
        parts.append(f'[KE.M]::mouse_event({self.M_MOVE}, {int(dx)}, {int(dy)}, 0, [UIntPtr]::Zero);')
        parts.append(f'Start-Sleep -Milliseconds {int(inter_ms)};')
        parts.append(f'[KE.M]::mouse_event({down}, 0, 0, 0, [UIntPtr]::Zero);')
        parts.append(f'Start-Sleep -Milliseconds {int(hold_ms)};')
        parts.append(f'[KE.M]::mouse_event({up}, 0, 0, 0, [UIntPtr]::Zero);')

        body = '\n'.join(parts)
        self._run_ps(body)

    def press(self, button: Literal['left', 'right', 'middle'] = 'left') -> None:
        """按下鼠标按钮（不释放）。"""
        if button == 'left':
            flag = self.M_LDOWN
        elif button == 'right':
            flag = self.M_RDOWN
        elif button == 'middle':
            flag = self.M_MDOWN
        else:
            raise ValueError('button must be left/right/middle')
        if _send_to_server(f'MOUSE_PRESS {button}') is not None:
            return
        body = f'[KE.M]::mouse_event({flag}, 0, 0, 0, [UIntPtr]::Zero);'
        self._run_ps(body)

    def release(self, button: Literal['left', 'right', 'middle'] = 'left') -> None:
        """释放鼠标按钮。"""
        if button == 'left':
            flag = self.M_LUP
        elif button == 'right':
            flag = self.M_RUP
        elif button == 'middle':
            flag = self.M_MUP
        else:
            raise ValueError('button must be left/right/middle')
        if _send_to_server(f'MOUSE_RELEASE {button}') is not None:
            return
        body = f'[KE.M]::mouse_event({flag}, 0, 0, 0, [UIntPtr]::Zero);'
        self._run_ps(body)

    def scroll(self, delta: int) -> None:
        """滚轮：正值向上，负值向下。通常 120 为一刻度。"""
        if _send_to_server(f'MOUSE_SCROLL {int(delta)}') is not None:
            return
        body = f'[KE.M]::mouse_event({self.M_WHEEL}, 0, 0, {int(delta)}, [UIntPtr]::Zero);'
        self._run_ps(body)

    def is_button_pressed(self, button: Literal['x1','x2','left','right','middle']='x2') -> bool:
        """检测指定鼠标按钮当前是否被按下（实时查询）。

        - `x1` / `x2` 对应鼠标侧键（XBUTTON1/XBUTTON2，Windows 虚拟键 0x05/0x06）。
        - 返回 True 表示当前为按下状态。
        注意：此功能通过 PowerShell 调用 `GetAsyncKeyState` 实现，会在 Windows 侧临时编译一次小型 C# 类型。
        """
        vk_map = {
            'x1': 0x05,
            'x2': 0x06,
            'left': 0x01,
            'right': 0x02,
            'middle': 0x04,
        }
        key = button.lower()
        if key not in vk_map:
            raise ValueError('button must be x1/x2/left/right/middle')
        vk = vk_map[key]

        # 使用 GetAsyncKeyState 检测高位位 0x8000
        # try server first
        resp = _send_to_server(f'IS_BUTTON {vk}', expect_reply=True)
        if resp is not None:
            return resp.strip().startswith('1')

        ps = (
            'Add-Type -TypeDefinition @"\n'
            'using System;\n'
            'using System.Runtime.InteropServices;\n'
            'public class K {\n'
            '    [DllImport("user32.dll")]\n'
            '    public static extern short GetAsyncKeyState(int vKey);\n'
            '}\n'
            '"@ -Language CSharp;\n'
            f'$s = [K]::GetAsyncKeyState({vk});\n'
            'if (($s -band 0x8000) -ne 0) { Write-Output 1 } else { Write-Output 0 }'
        )
        try:
            out = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", ps], text=True)
            return out.strip().startswith('1')
        except subprocess.CalledProcessError:
            return False


# 模块级默认实例与便捷函数
_DEFAULT_MOUSE = MouseController()

def mouse_move(dx: int, dy: int) -> None:
    _DEFAULT_MOUSE.move(dx, dy)

def mouse_click(button: Literal['left', 'right', 'middle'] = 'left', hold_ms: int | None = None) -> None:
    _DEFAULT_MOUSE.click(button, hold_ms)

def mouse_press(button: Literal['left', 'right', 'middle'] = 'left') -> None:
    _DEFAULT_MOUSE.press(button)

def mouse_release(button: Literal['left', 'right', 'middle'] = 'left') -> None:
    _DEFAULT_MOUSE.release(button)

def mouse_scroll(delta: int) -> None:
    _DEFAULT_MOUSE.scroll(delta)
