# [x]: Windows中的监听器操作
"""在 WSL 中启动 Windows 上的控制监听器（不使用独立 .ps1 文件）。

此脚本将把一个 PowerShell 服务器脚本编码为 Base64（UTF-16LE），并通过
`powershell.exe -Command "Start-Process powershell.exe -ArgumentList '-NoProfile','-EncodedCommand','<b64>' -WindowStyle Hidden'"`
在 Windows 上后台启动它，从而在 Windows 主机上监听本地端口。

用法：
  python start_win_listener.py        # 启动并等待就绪
  python start_win_listener.py --stop # 向服务器发送 SHUTDOWN

注意：需要 Windows 可用的 PowerShell（`powershell.exe` 在 PATH 中）。
"""

from __future__ import annotations
import base64
import subprocess
import socket
import time
import argparse
import sys


SERVER_PORT = 54321


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

Write-Host "Control server starting on port {port}"

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, {port})
$listener.Start()
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
    """Start/stop the Windows control listener from WSL.

    Usage:
        mgr = StartWinListener(port=54321)
        mgr.start()
        mgr.stop()
    """

    def __init__(self, port: int = SERVER_PORT, wait_ready: float = 5.0):
        self.port = int(port)
        self.wait_ready = float(wait_ready)

    @staticmethod
    def _encode_powershell(script: str) -> str:
        b = script.encode('utf-16le')
        return base64.b64encode(b).decode('ascii')

    def start(self) -> None:
        # inject port into PS_SERVER
        script = PS_SERVER.replace('{port}', str(self.port))
        b64 = self._encode_powershell(script)

        # start a background powershell that runs the encoded command
        outer_cmd = [
            'powershell.exe', '-NoProfile', '-Command',
            f"Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile','-EncodedCommand','{b64}' -WindowStyle Hidden"
        ]
        subprocess.run(outer_cmd, check=True)

        # wait until port ready
        deadline = time.time() + self.wait_ready
        while time.time() < deadline:
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=0.2) as s:
                    return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(f'Windows server not listening on 127.0.0.1:{self.port} after {self.wait_ready}s')

    def stop(self, timeout: float = 1.0) -> None:
        try:
            with socket.create_connection(('127.0.0.1', self.port), timeout=0.5) as s:
                s.sendall(b'SHUTDOWN\n')
                try:
                    s.recv(1024)
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f'Failed to contact server on port {self.port}: {e}')


# Backwards-compatible wrappers
def start_server(port: int = SERVER_PORT, wait_ready: float = 5.0) -> None:
    StartWinListener(port=port, wait_ready=wait_ready).start()


def stop_server(port: int = SERVER_PORT, timeout: float = 1.0) -> None:
    StartWinListener(port=port).stop(timeout=timeout)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--port', '-p', type=int, default=SERVER_PORT)
    p.add_argument('--stop', action='store_true')
    args = p.parse_args(argv)
    if args.stop:
        stop_server(args.port)
        print('Sent shutdown')
        return
    print('Starting Windows listener...')
    start_server(args.port)
    print(f'Windows listener started on 127.0.0.1:{args.port}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(1)
