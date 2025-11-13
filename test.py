#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _is_wsl() -> bool:
	try:
		with open("/proc/version", "r", encoding="utf-8") as f:
			ver = f.read()
		if "Microsoft" in ver or "microsoft" in ver:
			return True
	except Exception:
		pass
	if os.environ.get("WSL_DISTRO_NAME"):
		return True
	return False


def _now_filename() -> str:
	return datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")


def _diagnose_failure(exc: Exception) -> None:
	print("\n--- 环境诊断开始 ---")
	disp = os.environ.get("DISPLAY")
	way = os.environ.get("WAYLAND_DISPLAY")
	xauth = os.environ.get("XAUTHORITY")
	user = os.environ.get("USER") or os.environ.get("USERNAME") or "(unknown)"
	try:
		is_root = (os.geteuid() == 0)
	except Exception:
		is_root = False

	print(f"USER: {user}  (root={is_root})")
	print(f"DISPLAY: {disp!r}")
	print(f"WAYLAND_DISPLAY: {way!r}")
	print(f"XAUTHORITY: {xauth!r}")

	for cmd in ("xdpyinfo", "xrandr", "xprop", "xhost"):
		path = shutil.which(cmd)
		print(f"{cmd}: {'found at '+path if path else 'not found'}")

	print("\n错误摘要:")
	print(f"  {exc}")

	print("\n建议:")
	print("- 确保 Windows 侧有登录的桌面会话并未锁屏。")
	print("- 在 WSL 中避免 sudo 运行此脚本（不要改变 Windows 用户上下文）。")
	print("- 可在 Windows 侧直接运行 PowerShell 脚本以调试错误。")
	print("--- 环境诊断结束 ---\n")


def capture_windows_from_wsl(win_path: str = r"G:\\trans\\screenshot.png") -> str:
	"""Call Windows PowerShell from WSL to capture the full virtual screen and save to win_path.

	Returns the WSL path that maps to the saved file (e.g. /mnt/c/Users/Public/screenshot.png).
	Raises RuntimeError on failure with PowerShell output included.
	"""
	# choose available powershell binary name
	ps_bin = shutil.which("powershell.exe") or shutil.which("pwsh.exe") or "powershell.exe"

	# build single-line PowerShell command
	# it captures the virtual screen and saves as PNG
	ps_cmd = (
		'Add-Type -AssemblyName System.Windows.Forms,System.Drawing; '
		'$bmp = New-Object System.Drawing.Bitmap([System.Windows.Forms.SystemInformation]::VirtualScreen.Width, [System.Windows.Forms.SystemInformation]::VirtualScreen.Height); '
		'$g = [System.Drawing.Graphics]::FromImage($bmp); '
		'$g.CopyFromScreen([System.Drawing.Point]::Empty, [System.Drawing.Point]::Empty, $bmp.Size); '
		f'$bmp.Save("{win_path}",[System.Drawing.Imaging.ImageFormat]::Png); Write-Output "SAVED:{win_path}"'
	)

	args = [ps_bin, "-NoProfile", "-Command", ps_cmd]
	try:
		proc = subprocess.run(args, capture_output=True, text=True, check=False)
	except FileNotFoundError:
		raise RuntimeError("powershell.exe 未在 PATH 中，请在 WSL 中确保可以访问 powershell.exe（WSL 会自动映射 Windows 可执行文件）。")
	except Exception as e:
		raise RuntimeError(f"启动 PowerShell 时出错: {e}")

	stdout = (proc.stdout or "").strip()
	stderr = (proc.stderr or "").strip()

	if proc.returncode != 0:
		raise RuntimeError(f"PowerShell 命令失败 (rc={proc.returncode}). stdout={stdout!r} stderr={stderr!r}")

	# verify file exists via WSL path
	# convert Windows path like C:\\Users\\Public\\screenshot.png -> /mnt/c/Users/Public/screenshot.png
	wsl_path = win_path.replace("\\", "/")
	if len(wsl_path) >= 2 and wsl_path[1] == ":":
		drive = wsl_path[0].lower()
		rest = wsl_path[2:]
		if rest.startswith("/"):
			rest = rest[1:]
		wsl_path = f"/mnt/{drive}/{rest}"

	if not Path(wsl_path).exists():
		raise RuntimeError(f"截图已在 Windows 保存，但在 WSL 路径未找到: {wsl_path}. PowerShell 输出: stdout={stdout!r} stderr={stderr!r}")

	return wsl_path

def main(argv: Optional[list[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="在 WSL 中调用 Windows 截图并把文件读取到 WSL")
	parser.add_argument("--win-path", help=r"Windows 保存路径，例如 G:\\trans\\screenshot.png", default=r"G:\\trans\\screenshot.png")
	parser.add_argument("--output", help="WSL 端另存为路径（可选），例如 /tmp/s1.png; 若不指定则显示 Windows 对应的 /mnt 路径")
	parser.add_argument("--base64", action="store_true", help="将图片以 base64 打印到 stdout（不保存到 output）")
	parser.add_argument("--check-wsl", action="store_true", help="仅检查是否在 WSL 环境")
	args = parser.parse_args(argv)

	if args.check_wsl:
		print("is_wsl=", _is_wsl())
		return 0

	if not _is_wsl():
		print("警告: 当前看起来并非在 WSL 环境运行；脚本仍会尝试调用 powershell.exe。")

	try:
		wsl_saved = capture_windows_from_wsl(args.win_path)
	except Exception as e:
		print("截屏失败：", e, file=sys.stderr)
		try:
			_diagnose_failure(e)
		except Exception:
			pass
		return 1

	# 如果要求 base64 输出
	if args.base64:
		try:
			data = Path(wsl_saved).read_bytes()
			b64 = base64.b64encode(data).decode('ascii')
			print(b64)
			return 0
		except Exception as e:
			print("读取或编码图片失败：", e, file=sys.stderr)
			return 2

	# 如果指定 --output，把文件复制到指定位置（WSL 路径）
	if args.output:
		try:
			outp = Path(args.output)
			outp.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(wsl_saved, outp)
			print(f"已保存到 WSL 路径: {outp}")
			return 0
		except Exception as e:
			print("复制到输出失败：", e, file=sys.stderr)
			return 3

	# 否则打印默认 WSL 路径
	print(f"已在 Windows 保存，WSL 可访问路径: {wsl_saved}")
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

