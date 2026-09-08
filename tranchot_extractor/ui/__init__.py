"""
User Interface module for Tranchot Extractor.
"""

def __getattr__(name):
    if name in ("create_app", "launch"):
        from tranchot_extractor.ui.app import create_app, launch
        return create_app if name == "create_app" else launch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["create_app", "launch"]
