"""Exercise the unmodified production GUI path for a bounded number of frames."""
import argparse
import json
from pathlib import Path
import time

import taichi as ti
from project.main import run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_window = ti.ui.Window
    windows = []
    counters = {'frames_presented': 0, 'get_gui_calls': 0}

    class BoundedWindow:
        def __init__(self, *positional, **keywords):
            self.actual = original_window(*positional, **keywords)
            windows.append(self.actual)

        @property
        def running(self):
            return self.actual.running and counters['frames_presented'] < args.frames

        def get_gui(self):
            counters['get_gui_calls'] += 1
            return self.actual.get_gui()

        def show(self):
            # save_image renders the queued canvas/GUI; show presents the real window.
            if counters['frames_presented'] == args.frames - 1:
                self.actual.save_image(str(args.output_dir / 'ggui.png'))
            self.actual.show()
            counters['frames_presented'] += 1

        def __getattr__(self, name):
            return getattr(self.actual, name)

    ti.ui.Window = BoundedWindow
    started = time.perf_counter()
    result = {'solver_requested': 'cuda', 'production_entry': 'project.main.run_demo',
              'scope': 'Real window creation, GUI widgets and canvas draw/present; no manual clicking of each control',
              'passed': False}
    try:
        run_demo(arch='cuda')
        result.update(counters)
        result['actual_solver_arch'] = str(ti.lang.impl.current_cfg().arch)
        result['passed'] = counters['frames_presented'] == args.frames and counters['get_gui_calls'] == args.frames
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        ti.ui.Window = original_window
        for window in windows:
            window.destroy()
        result['elapsed_seconds'] = time.perf_counter() - started
        (args.output_dir / 'ggui_probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit('GUI closed before all validation frames were presented')


if __name__ == '__main__':
    main()
