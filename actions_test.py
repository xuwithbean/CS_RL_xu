from __future__ import annotations

import argparse
import time

from actions import m_actions


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="m_actions test script")
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "x2-loop"],
        help="basic: 基础动作串测；x2-loop: 循环动作，按鼠标侧键 x2 中断",
    )
    parser.add_argument("--loop-interval", type=float, default=0.05, help="x2-loop 模式每轮间隔秒数")
    parser.add_argument("--move-dx", type=int, default=200, help="x2-loop 模式每轮鼠标水平移动")
    parser.add_argument("--move-dy", type=int, default=0, help="x2-loop 模式每轮鼠标垂直移动")
    parser.add_argument("--click-hold-sec", type=float, default=0.05, help="点击按住时长（秒）")
    return parser.parse_args()


def run_basic_test(act: m_actions) -> None:
    print("[actions_test] basic test start", flush=True)

    # 键盘动作
    act.move_forward(0.1)
    act.move_left(0.1)
    act.move_right(0.1)
    act.move_back(0.1)
    act.jump(0.05)
    act.reload(0.05)
    act.switch_primary_weapon(0.05)
    act.switch_secondary_weapon(0.05)
    act.switch_knife(0.05)
    act.crouch(0.1)
    act.crouch_end()

    # 鼠标动作
    act.mouse_move(80, 0)
    act.mouse_click(hold_sec=0.05)
    act.mouse_move_click(60, 0, hold_sec=0.05)
    act.mouse_click_interval(click_times=2, interval_sec=0.1, hold_sec=0.05)

    act.stop()
    print("[actions_test] basic test done", flush=True)


def run_x2_loop_test(act: m_actions, loop_interval: float, move_dx: int, move_dy: int, click_hold_sec: float) -> None:
    print("[actions_test] x2-loop start", flush=True)
    print("[actions_test] press mouse side button x2 to stop", flush=True)

    rounds = 0
    try:
        while True:
            if act.stop_if_interrupt_x2():
                print("[actions_test] x2 detected, loop stopped", flush=True)
                break

            act.mouse_move_click(move_dx, move_dy, hold_sec=click_hold_sec)
            rounds += 1

            if rounds % 20 == 0:
                print(f"[actions_test] rounds={rounds}", flush=True)

            time.sleep(max(0.0, float(loop_interval)))
    finally:
        act.stop()


def main() -> int:
    args = get_args()
    act = m_actions()

    if args.mode == "basic":
        run_basic_test(act)
        return 0

    if args.mode == "x2-loop":
        run_x2_loop_test(
            act,
            loop_interval=args.loop_interval,
            move_dx=args.move_dx,
            move_dy=args.move_dy,
            click_hold_sec=args.click_hold_sec,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
