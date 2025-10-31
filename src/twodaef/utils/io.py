from pathlib import Path
import joblib

def ensure_dir(path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

def save_joblib(obj, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)

def load_joblib(path: str | Path):
    return joblib.load(path)
