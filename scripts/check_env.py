"""
check_env.py - Verify that all required libraries are importable.
Prints "OK: <library>" or "MISSING: <library>" for each dependency.
Never crashes on import errors.
"""
from __future__ import annotations

LIBRARIES = [
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("librosa", "librosa"),
    ("jiwer", "jiwer"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pydantic", "pydantic"),
    ("faster_whisper", "faster-whisper"),
    ("pyarrow", "pyarrow"),
    ("miniaudio", "miniaudio"),
    ("pytest", "pytest"),
]


def check_library(import_name: str, display_name: str) -> None:
    try:
        __import__(import_name)
        print(f"OK: {display_name}")
    except ImportError as exc:
        print(f"MISSING: {display_name}  ({exc})")
    except Exception as exc:
        print(f"MISSING: {display_name}  (unexpected error: {exc})")


def main() -> None:
    print("=" * 50)
    print("Quranic Pipeline – Environment Check")
    print("=" * 50)
    for display_name, import_name in LIBRARIES:
        check_library(import_name, display_name)
    print("=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
