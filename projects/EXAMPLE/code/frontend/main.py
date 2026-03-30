import io
import sys
import os
import base64
import traceback
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — MUST be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── project-root on sys.path so `models.*` imports work ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.preparing import Dataset
from models.modeling import ModelTrainer
from models.visualizing import EDAVisualizer, ModelVisualizer

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Pipeline UI")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory store for uploaded files keyed by session token
_UPLOADS: dict[str, pd.DataFrame] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def capture_stdout():
    """Redirect stdout to a StringIO buffer; yield the buffer."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    # Higher DPI keeps labels readable once rendered in the browser.
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=170)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _patch_plt_show(figures: list):
    """
    Monkey-patch plt.show() so that instead of displaying a figure it
    captures the current figure into `figures` as a base64 PNG.
    Returns a restore function.
    """
    original_show = plt.show

    def _fake_show(*args, **kwargs):
        fig = plt.gcf()
        if fig.get_axes():                    # only capture if there's something drawn
            figures.append(_fig_to_b64(fig))
            plt.close(fig)

    plt.show = _fake_show
    return original_show


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV file and return its columns + a preview of the first 5 rows."""
    filename = (file.filename or "").strip().lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    token = str(uuid.uuid4())
    _UPLOADS[token] = df

    # Build a preview with JSON-safe values: NaN/Inf -> None
    preview_df = df.head(5).replace([np.inf, -np.inf], np.nan).astype(object)
    preview = preview_df.where(pd.notnull(preview_df), None).values.tolist()

    return JSONResponse({
        "token": token,
        "columns": df.columns.tolist(),
        "shape": list(df.shape),
        "preview_columns": df.columns.tolist(),
        "preview_rows": preview,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    })


# ── Run request schema ────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    token: str
    target: str
    features: Optional[List[str]] = None       # None = all non-target, non-ts columns
    timestamp_col: Optional[str] = None
    is_time_series: bool = False
    is_classification: bool = False
    shuffle: bool = True
    test_size: float = 0.2
    val_size: Optional[float] = None
    n_trials: int = 20
    scale_numeric: bool = False
    use_knn_for_numeric: bool = False


@app.post("/run")
async def run_pipeline(req: RunRequest):
    """
    Run the full pipeline:
      1. Dataset construction + EDA visualisation
      2. dataset.prepare_dataset()
      3. ModelTrainer (Optuna)
      4. ModelVisualizer
    Returns logs, EDA plots, model metrics and model plots — all as JSON.
    """
    if req.token not in _UPLOADS:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload again.")

    df = _UPLOADS[req.token].copy()

    # Validation
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target}' not found in dataset.")
    if req.features:
        missing = [c for c in req.features if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Feature columns not found: {missing}")
    if req.timestamp_col and req.timestamp_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Timestamp column '{req.timestamp_col}' not found.")

    # Auto-encode string target to integers for classification
    if req.is_classification and not pd.api.types.is_numeric_dtype(df[req.target]):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[req.target] = le.fit_transform(df[req.target])

    # Force no shuffle for time series
    shuffle = False if req.is_time_series else req.shuffle

    logs: list[str] = []
    eda_plots: list[str] = []
    model_plots: list[str] = []
    metrics: dict = {}

    original_show = _patch_plt_show(eda_plots)   # intercept plt.show() for EDA phase

    try:
        # ── 1. Build Dataset ──────────────────────────────────────────────────
        with capture_stdout() as buf:
            if req.timestamp_col:
                dataset = Dataset(
                    data=df,
                    target=req.target,
                    features=req.features,
                    timestamp_col=req.timestamp_col,
                )
            else:
                # Inject a dummy timestamp column if none provided
                df = df.copy()
                df["_ts"] = pd.date_range("2000-01-01", periods=len(df), freq="D")
                dataset = Dataset(
                    data=df,
                    target=req.target,
                    features=req.features,
                    timestamp_col="_ts",
                )
        logs.append(buf.getvalue())

        # ── 2. EDA Visualiser ─────────────────────────────────────────────────
        with capture_stdout() as buf:
            eda = EDAVisualizer(
                df=dataset.data,
                target=req.target,
                timestamp_col=req.timestamp_col or "_ts",
            )
            eda.run_all()
        logs.append(buf.getvalue() or "[EDA] Plots generated.")

        # Switch plt.show() interception to model_plots list
        plt.show = lambda *a, **kw: (
            model_plots.append(_fig_to_b64(plt.gcf())) or plt.close(plt.gcf())
            if plt.gcf().get_axes() else None
        )

        # ── 3. Prepare dataset ────────────────────────────────────────────────
        with capture_stdout() as buf:
            dataset.prepare_dataset(
                test_size=req.test_size,
                val_size=req.val_size,
                shuffle=shuffle,
                scale_numeric=req.scale_numeric,
                use_knn_for_numeric=req.use_knn_for_numeric,
            )
        logs.append(buf.getvalue() or "[Prep] Dataset prepared and split.")

        # ── 4. Model Training (Optuna) ────────────────────────────────────────
        with capture_stdout() as buf:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            trainer = ModelTrainer(
                dataset=dataset,
                is_classification=req.is_classification,
                n_trials=req.n_trials,
                shuffle=shuffle,
            )
        logs.append(buf.getvalue())

        # ── 5. Collect metrics ────────────────────────────────────────────────
        metrics["model_name"] = type(trainer.model).__name__
        metrics["best_params"] = {k: str(v) for k, v in trainer.best_params.items()}

        y_test = dataset.splits.y_test
        y_pred = trainer.y_test_pred
        if len(y_test) > 0:
            if req.is_classification:
                from sklearn.metrics import accuracy_score, classification_report
                metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
                metrics["report"] = classification_report(y_test, y_pred)
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                import math
                mse = mean_squared_error(y_test, y_pred)
                metrics["rmse"] = round(math.sqrt(mse), 6)
                metrics["r2"] = round(float(r2_score(y_test, y_pred)), 6)

        # ── 6. Model visualisation ────────────────────────────────────────────
        with capture_stdout() as buf:
            viz = ModelVisualizer(trainer)
            viz.run_all()
        logs.append(buf.getvalue() or "[ModelViz] Plots generated.")

    except Exception as exc:
        plt.show = original_show
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": tb},
        )
    finally:
        plt.show = original_show

    return JSONResponse({
        "logs": "\n".join(filter(None, logs)),
        "eda_plots": eda_plots,
        "model_plots": model_plots,
        "metrics": metrics,
    })
