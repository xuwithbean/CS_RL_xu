"""动作组合封装，方便后续在训练、推理或脚本里直接调用。"""

import time

from control import KeySender
from control import MouseController


mymouse = MouseController()
mykey = KeySender()


class m_actions:
    def __init__(self, key_sender=None, mouse_controller=None):
        self.key_sender = key_sender or mykey
        self.mouse_controller = mouse_controller or mymouse

    def _press_or_hold(self, key, hold_sec=None):
        if hold_sec is None:
            self.key_sender.press(key)
            return
        self.key_sender.press_and_release(key, hold_ms=int(float(hold_sec) * 1000))

    def move_forward(self, hold_sec=None):
        self._press_or_hold('w', hold_sec)

    def move_back(self, hold_sec=None):
        self._press_or_hold('s', hold_sec)

    def move_left(self, hold_sec=None):
        self._press_or_hold('a', hold_sec)

    def move_right(self, hold_sec=None):
        self._press_or_hold('d', hold_sec)

    def jump(self, hold_sec=None):
        self._press_or_hold('space', hold_sec)

    def reload(self, hold_sec=None):
        self._press_or_hold('r', hold_sec)

    def switch_knife(self, hold_sec=None):
        self._press_or_hold('4', hold_sec)

    def switch_primary_weapon(self, hold_sec=None):
        self._press_or_hold('1', hold_sec)

    def switch_secondary_weapon(self, hold_sec=None):
        self._press_or_hold('2', hold_sec)

    def crouch(self, hold_sec=None):
        self._press_or_hold('ctrl', hold_sec)

    def crouch_end(self, hold_sec=None):
        self.key_sender.release('ctrl')

    def show_scoreboard(self, hold_sec=None):
        self._press_or_hold('tab', hold_sec)

    def open_buy_menu(self, hold_sec=None):
        self._press_or_hold('b', hold_sec)

    def interact(self, hold_sec=None):
        self._press_or_hold('e', hold_sec)

    def mouse_move(self, dx, dy):
        self.mouse_controller.move(int(dx), int(dy))

    def mouse_click(self, hold_sec=None):
        hold_ms = 50 if hold_sec is None else int(float(hold_sec) * 1000)
        self.mouse_controller.click('left', hold_ms=hold_ms)

    def mouse_click_interval(self, click_times=2, interval_sec=0.1, hold_sec=None):
        click_times = max(1, int(click_times))
        interval_sec = max(0.0, float(interval_sec))
        hold_ms = 50 if hold_sec is None else int(float(hold_sec) * 1000)
        for idx in range(click_times):
            if self.is_interrupt_x2_pressed():
                return False
            self.mouse_controller.click('left', hold_ms=hold_ms)
            if idx < click_times - 1:
                time.sleep(interval_sec)
        return True

    def mouse_hold_left(self):
        self.mouse_controller.press('left')

    def mouse_release_left(self):
        self.mouse_controller.release('left')

    def mouse_move_click(self, dx, dy, hold_sec=None):
        self.mouse_move(dx, dy)
        self.mouse_click(hold_sec=hold_sec)

    def mouse_move_click_interval(self, dx, dy, click_times=2, interval_sec=0.1, hold_sec=None):
        if self.is_interrupt_x2_pressed():
            return False
        self.mouse_move(dx, dy)
        return self.mouse_click_interval(click_times=click_times, interval_sec=interval_sec, hold_sec=hold_sec)

    def mouse_move_hold_left(self, dx, dy):
        self.mouse_move(dx, dy)
        self.mouse_hold_left()

    def stop(self):
        self.key_sender.release('w')
        self.key_sender.release('s')
        self.key_sender.release('a')
        self.key_sender.release('d')
        self.mouse_release_left()

    def is_interrupt_x2_pressed(self):
        """检测鼠标侧键 X2 是否按下，用于外部循环中断。"""
        try:
            return bool(self.mouse_controller.is_button_pressed('x2'))
        except Exception:
            return False

    def stop_if_interrupt_x2(self):
        """若检测到 X2，则执行 stop 并返回 True。"""
        if self.is_interrupt_x2_pressed():
            self.stop()
            return True
        return False

    def wait(self, sec=0.1):
        time.sleep(float(sec))