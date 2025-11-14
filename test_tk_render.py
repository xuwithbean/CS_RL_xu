"""
Simple Tkinter image render test.

Usage:
  python test_tk_render.py /mnt/g/trans/screenshot.png
If the file exists it will try to load it (via PIL if available or Tk PhotoImage fallback).
If no file is provided or file doesn't exist, the script will generate a simple test image (requires Pillow) or draw a colored rectangle on a Canvas.

This script prints diagnostics ($DISPLAY/$WAYLAND_DISPLAY) and any load errors.
"""

import os
import sys
import argparse
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default=None, help='Path to PNG image to display')
    args = parser.parse_args()

    print('[test_tk_render] DISPLAY=%s WAYLAND_DISPLAY=%s' % (os.environ.get('DISPLAY'), os.environ.get('WAYLAND_DISPLAY')))

    try:
        import tkinter as tk
    except Exception as e:
        print('Tkinter import failed:', e)
        sys.exit(2)

    use_pil = False
    try:
        from PIL import Image, ImageTk, ImageDraw
        use_pil = True
    except Exception:
        Image = None
        ImageTk = None
        ImageDraw = None

    root = tk.Tk()
    root.title('Tk Render Test')

    # Try to set a reasonable default size
    win_w, win_h = 800, 600
    try:
        root.geometry(f'{win_w}x{win_h}')
    except Exception:
        pass

    frame = tk.Frame(root)
    frame.pack(expand=True, fill='both')

    label = tk.Label(frame)
    label.pack(expand=True, fill='both')

    loaded_img_ref = None

    def show_with_pil(path):
        nonlocal loaded_img_ref
        try:
            img = Image.open(path)
            ow, oh = img.size
            # scale to window size
            w = max(1, label.winfo_width() or win_w)
            h = max(1, label.winfo_height() or win_h)
            try:
                img = img.resize((w, h), Image.LANCZOS)
            except Exception:
                pass
            tkimg = ImageTk.PhotoImage(img)
            label.config(image=tkimg)
            label.image = tkimg
            loaded_img_ref = tkimg
            print(f'[test_tk_render] loaded via PIL: original=({ow},{oh}) display=({w},{h})')
            return True
        except Exception:
            print('[test_tk_render] PIL failed to load/display image:')
            traceback.print_exc()
            return False

    def show_with_tk(path):
        nonlocal loaded_img_ref
        try:
            tkimg = tk.PhotoImage(file=path)
            label.config(image=tkimg)
            label.image = tkimg
            loaded_img_ref = tkimg
            print('[test_tk_render] loaded via tk.PhotoImage')
            return True
        except Exception:
            print('[test_tk_render] tk.PhotoImage failed:')
            traceback.print_exc()
            return False

    def draw_canvas_test():
        canvas = tk.Canvas(frame, bg='black')
        canvas.pack(expand=True, fill='both')
        # draw gradient-like rectangles
        try:
            w = canvas.winfo_width() or win_w
            h = canvas.winfo_height() or win_h
            for i in range(0, 10):
                color = '#%02x%02x%02x' % (25 * i, 255 - 25 * i, 150)
                canvas.create_rectangle(i * (w // 10), 0, (i + 1) * (w // 10), h, fill=color, outline='')
            canvas.create_text(10, 10, anchor='nw', text='Canvas fallback: no image available', fill='white')
            print('[test_tk_render] drew fallback canvas')
        except Exception:
            traceback.print_exc()

    def ensure_and_display():
        # Called after mainloop starts so widget sizes are available
        path = args.image
        if path and os.path.exists(path):
            print('[test_tk_render] image file exists:', path)
            if use_pil:
                ok = show_with_pil(path)
                if not ok:
                    show_with_tk(path)
            else:
                ok = show_with_tk(path)
                if not ok:
                    draw_canvas_test()
        else:
            print('[test_tk_render] image not provided or not found; generating test content')
            if use_pil:
                # generate a simple image in memory and display
                try:
                    img = Image.new('RGB', (win_w, win_h), color=(30, 30, 30))
                    d = ImageDraw.Draw(img)
                    d.text((10, 10), 'Generated test image (PIL)', fill=(220, 220, 220))
                    tkimg = ImageTk.PhotoImage(img)
                    label.config(image=tkimg)
                    label.image = tkimg
                    loaded_img_ref = tkimg
                    print('[test_tk_render] displayed generated PIL image')
                except Exception:
                    traceback.print_exc()
                    draw_canvas_test()
            else:
                draw_canvas_test()

    # schedule initial display
    root.after(200, ensure_and_display)

    root.mainloop()


if __name__ == '__main__':
    main()
