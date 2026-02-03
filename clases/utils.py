from datetime import datetime
from pathlib import Path
import contextlib
import json
import os
import sys
import keras


class Utils:
    @staticmethod
    def ensure_dir(p: Path) -> None:
        Path(p).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    @staticmethod
    def save_json(obj: dict, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        keras.utils.set_random_seed(seed)

    @staticmethod
    @contextlib.contextmanager
    def suppress_stdout_stderr():
        """Silencia stdout/stderr temporalmente (útil para TFLite converter/MLIR)."""
        with open(os.devnull, "w") as devnull:
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = devnull, devnull
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err




