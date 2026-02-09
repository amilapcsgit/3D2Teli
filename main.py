import os
import sys

def run_tk_fallback() -> None:
    from gui import FlattenApp

    app = FlattenApp()
    app.mainloop()


def run_default_app() -> int:
    force_tk = os.environ.get("TENTMAKER_FORCE_TK", "").strip() == "1"
    if force_tk:
        run_tk_fallback()
        return 0

    try:
        from qt_app.main_window import run_qt_app

        return int(run_qt_app())
    except Exception as exc:
        print(f"[Tent-Maker] Qt launch failed, falling back to Tkinter: {exc}")
        run_tk_fallback()
        return 0


if __name__ == "__main__":
    raise SystemExit(run_default_app())
