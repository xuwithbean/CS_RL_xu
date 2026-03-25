from keycontrol import KeySender
from mousecontrol import MouseController
mykey=KeySender()
mymouse = MouseController()
print('Starting loop. Press mouse side button 2 (XButton2 / mouse button 5) to stop.')
try:
    while True:
        # 合并为一次调用以减少延迟
        mykey.press('w')
        mymouse.move_and_click(1200, 0, button='left', hold_ms=10, inter_ms=5)
        # 快速检测是否按下鼠标侧键（XBUTTON2，mouse button 5）
        if mymouse.is_button_pressed('x2'):
            print('Stop button detected; exiting.')
            break
        # 小睡一下以避免占满 CPU（可根据需要调整）
except KeyboardInterrupt:
    print('Interrupted by user')