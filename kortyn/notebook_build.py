"""Build ``notebooks/gpu_benchmark.ipynb`` (phase 5) from its parts.

The notebook must run unchanged in Google Colab and on Kaggle, without access to this repository. It therefore
embeds the measurement script (:mod:`kortyn.notebook_bench`, verbatim) and eight LIBERO-Long observations
(``notebooks/libero_observations.json``). Its own cells only check the GPU, install a pinned Python 3.12
environment with ``uv``, start the script and offer the result file for download.
``tests/test_notebook.py`` checks that the committed notebook equals this build.

    python -m kortyn.notebook_build            # writes notebooks/gpu_benchmark.ipynb
    python -m kortyn.notebook_build --check    # exit 1 if the committed notebook is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "gpu_benchmark.ipynb"
OBSERVATIONS = ROOT / "notebooks" / "libero_observations.json"
SCRIPT = ROOT / "kortyn" / "notebook_bench.py"
REPOSITORY = "https://github.com/erithy25/Kortyn-Robotics"
GUIDE = f"{REPOSITORY}/blob/phase-5-publish/docs/gpu_benchmark_browser.md"

UV_VERSION = "0.11.32"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
TORCH_PACKAGES = ("torch==2.11.0", "torchvision==0.26.0")
PACKAGES = (
    "lerobot[smolvla]==0.6.1",
    "transformers==5.5.4",
    "huggingface-hub==1.33.0",
    "numpy==2.2.6",
    "safetensors==0.8.0",
    "accelerate==1.15.0",
    "tokenizers==0.22.2",
    "num2words==0.5.14",
    "pillow==12.3.0",
)


class BuildError(RuntimeError):
    """The parts cannot be embedded (missing file, text a raw string cannot hold)."""


INTRO = f"""# Kortyn: SmolVLA auf der GPU messen / measure SmolVLA on a GPU

Dieses Notebook misst, wie lange das Robotermodell **SmolVLA** (`lerobot/smolvla_libero`) auf der GPU dieser \
Sitzung für eine Anfrage braucht, einzeln und gebündelt (Batch 1–32), in drei Genauigkeiten (fp32, fp16, wie \
geladen) und kompiliert. Das Ergebnis ist eine JSON-Datei, die Kortyn direkt einliest.

**So geht's (Anleitung mit Bildern: [docs/gpu_benchmark_browser.md]({GUIDE})):**

1. **GPU einschalten.** Colab: Menü *Laufzeit → Laufzeittyp ändern → T4-GPU → Speichern*. Kaggle: rechts \
*Session options → Accelerator → GPU T4 x2*, *Internet* an.
2. **Alles ausführen.** Colab: *Laufzeit → Alle ausführen*. Kaggle: *Run All*. Dauer: Installation \
≈ 5 Minuten, Messung ≈ 30–60 Minuten. Den Tab offen lassen.
3. **Datei herunterladen** (letzter Schritt unten) und im Repository unter `results/phase5/gpu/` hochladen.

Ohne GPU bricht das Notebook im ersten Schritt mit einer Anleitung ab. Es sendet keine Telemetrie; geladen \
werden nur Pakete (PyPI, PyTorch) und das Modell (Hugging Face). Die Messdatei bleibt bei dir, bis du sie \
hochlädst.

---

*English:* enable a GPU runtime (Colab: Runtime → Change runtime type → T4 GPU; Kaggle: Accelerator → \
GPU T4 x2, Internet on), run all cells (≈ 5 min install, 30–60 min measurement), download the JSON file at \
the end and add it to `results/phase5/gpu/` in [{REPOSITORY}]({REPOSITORY}). Pre-registered protocol: \
`reports/phase5_preregistration.md`, section 4."""

GPU_CHECK = r'''#@title Schritt 1 – GPU prüfen / check the GPU
import os
import shutil
import subprocess
import sys
from pathlib import Path

ON_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir()
PLATFORM = "colab" if ON_COLAB else ("kaggle" if ON_KAGGLE else "other")
WORK_DIR = Path("/content" if ON_COLAB else ("/kaggle/working" if ON_KAGGLE else ".")).resolve()
ALLOW_CPU = os.environ.get("KORTYN_BENCHMARK_ALLOW_CPU") == "1"  # only for Kortyn's own test on a CPU machine

HELP = {
    "colab": (
        "Es ist keine GPU aktiv. So schaltest du sie in Google Colab ein:\n"
        "  1. Oben im Menü „Laufzeit“ → „Laufzeittyp ändern“.\n"
        "  2. Unter „Hardwarebeschleuniger“ „T4-GPU“ wählen → „Speichern“.\n"
        "  3. Menü „Laufzeit“ → „Alle ausführen“.\n"
        "Ist gerade keine kostenlose GPU frei, meldet Colab das; dann später noch einmal versuchen.\n\n"
        "No GPU is active. Runtime → Change runtime type → T4 GPU → Save, then Runtime → Run all."
    ),
    "kaggle": (
        "Es ist keine GPU aktiv. So schaltest du sie auf Kaggle ein:\n"
        "  1. Rechts im Bereich „Session options“ unter „Accelerator“ „GPU T4 x2“ wählen\n"
        "     (Kaggle verlangt dafür einmalig eine Telefon-Verifizierung des Kontos).\n"
        "  2. Ebenda „Internet“ einschalten (für die Installation).\n"
        "  3. Oben „Run All“ klicken.\n\n"
        "No GPU is active. Session options → Accelerator → GPU T4 x2, Internet on, then Run All."
    ),
    "other": (
        "Es ist keine NVIDIA-GPU zu sehen (nvidia-smi fehlt oder meldet keine Karte).\n"
        "Dieses Notebook ist für Google Colab oder Kaggle mit eingeschalteter GPU gedacht.\n\n"
        "No NVIDIA GPU visible (nvidia-smi missing or reports no card). Use Colab or Kaggle with a GPU."
    ),
}


class KeineGPU(RuntimeError):
    """No GPU active: the notebook stops here (Run all stops at this error)."""


def find_gpus() -> list[str]:
    """``name, memory, driver`` per NVIDIA GPU from nvidia-smi, or [] without one."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return []
    try:
        result = subprocess.run(
            [tool, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


GPUS = find_gpus()
if GPUS:
    print("GPU gefunden / GPU found:")
    for gpu in GPUS:
        print("  " + gpu)
elif ALLOW_CPU:
    print("Keine GPU: Testmodus auf der CPU (KORTYN_BENCHMARK_ALLOW_CPU=1), nur für Kortyns eigenen Test.")
else:
    print(HELP[PLATFORM])
    raise KeineGPU(
        "Keine GPU aktiv – bitte wie oben beschrieben einschalten und neu starten. "
        "/ No GPU active – enable one as described above and run all cells again."
    )'''

INSTALL = r"""#@title Schritt 2 – Installation (≈ 5 Minuten) / installation (≈ 5 minutes)
import os
import subprocess
import sys
import time
from pathlib import Path

UV_VERSION = "{uv}"
TORCH_INDEX = os.environ.get("KORTYN_TORCH_INDEX", "{index}")
TORCH_PACKAGES = {torch}
PACKAGES = {packages}
VENV = Path(os.environ.get("KORTYN_BENCHMARK_VENV", "/tmp/kortyn-venv"))
PY = VENV / "bin" / "python"
UV = [sys.executable, "-m", "uv"]


def step(command: list[str], what: str) -> None:
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(
            f"Installation fehlgeschlagen bei: {{what}}. Bitte „Alle ausführen“ noch einmal starten; "
            f"bleibt der Fehler, die Ausgabe oben kopieren. / Installation failed at: {{what}}."
        )
    print(f"  {{what}}: ok ({{time.time() - started:.0f}} s)")


print("Installiere eine eigene Python-3.12-Umgebung mit festen Versionen ...")
step([sys.executable, "-m", "pip", "install", "--quiet", f"uv=={{UV_VERSION}}"], "uv")
step([*UV, "venv", "--quiet", "--managed-python", "--python", "3.12", str(VENV)], "Python 3.12")
step(
    [*UV, "pip", "install", "--quiet", "--python", str(PY), "--index-url", TORCH_INDEX, *TORCH_PACKAGES],
    "PyTorch",
)
step([*UV, "pip", "install", "--quiet", "--python", str(PY), *PACKAGES], "LeRobot und SmolVLA")
check = subprocess.run(
    [str(PY), "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
    capture_output=True,
    text=True,
    check=False,
)
print("  torch, CUDA, GPU sichtbar / visible:", check.stdout.strip() or check.stderr.strip()[-2000:])
if "True" not in check.stdout and not ALLOW_CPU:
    raise RuntimeError(
        "PyTorch sieht die GPU nicht (Treiber oder Laufzeit). Bitte Laufzeit neu starten und erneut "
        "ausführen. / PyTorch does not see the GPU: restart the runtime and run all cells again."
    )"""

FILES = r"""#@title Messprogramm und Daten (nicht ändern) / script and inputs {{ display-mode: "form" }}
from pathlib import Path

BENCH_DIR = Path("/tmp/kortyn-bench")
BENCH_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT = r''{q}{script}''{q}
OBSERVATIONS = r''{q}{observations}''{q}
(BENCH_DIR / "kortyn_bench.py").write_text(SCRIPT, encoding="utf-8")
(BENCH_DIR / "observations.json").write_text(OBSERVATIONS, encoding="utf-8")
print("Messprogramm und Eingaben bereit / ready:", BENCH_DIR)"""

RUN = r"""#@title Schritt 3 – Messung (≈ 30–60 Minuten) / measurement (≈ 30–60 minutes)
import os
import subprocess

command = [
    str(PY), "-u", str(BENCH_DIR / "kortyn_bench.py"),
    "--observations", str(BENCH_DIR / "observations.json"),
    "--out-dir", str(WORK_DIR),
    "--platform", PLATFORM,
]
if ALLOW_CPU and not GPUS:  # Kortyn's CPU test of this notebook: few, small measurements
    command += [
        "--allow-cpu", "--device", "cpu",
        "--sizes", os.environ.get("KORTYN_BENCHMARK_SIZES", "1,2"),
        "--repeats", os.environ.get("KORTYN_BENCHMARK_REPEATS", "3"),
        "--warmup", os.environ.get("KORTYN_BENCHMARK_WARMUP", "1"),
        "--configs", os.environ.get("KORTYN_BENCHMARK_CONFIGS", "standard,fp32"),
    ]
environment = dict(
    os.environ,
    HF_HUB_DISABLE_TELEMETRY="1",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
    WANDB_MODE="disabled",
    TOKENIZERS_PARALLELISM="false",
    PYTHONUNBUFFERED="1",
    PYTHONWARNINGS="ignore",
)
before = set(WORK_DIR.glob("smolvla_gpu_*.json"))
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment
)
assert process.stdout is not None
for line in process.stdout:
    print(line, end="", flush=True)
code = process.wait()
written = sorted(set(WORK_DIR.glob("smolvla_gpu_*.json")) - before, key=lambda path: path.stat().st_mtime)
RESULT = written[-1] if written else None
if code != 0:
    hint = f" Bis dahin Gemessenes steht in {RESULT}." if RESULT else ""
    raise RuntimeError(
        f"Die Messung ist mit Fehler {code} abgebrochen (Ausgabe oben).{hint} / The run failed."
    )
if RESULT is None:
    raise RuntimeError("Die Messung hat keine Datei geschrieben (Ausgabe oben). / No result file written.")"""

DOWNLOAD = r"""#@title Schritt 4 – Ergebnis herunterladen / download the result
print("Ergebnisdatei / result file:", RESULT)
if ON_COLAB:
    from google.colab import files

    files.download(str(RESULT))
    print("Der Download startet im Browser (falls der Browser fragt: erlauben).")
elif ON_KAGGLE:
    from IPython.display import FileLink, display

    print("Kaggle: rechts unter „Output“ → /kaggle/working → Datei anklicken → ⋮ → „Download“.")
    display(FileLink(RESULT.name))
else:
    print("Die Datei liegt im Arbeitsverzeichnis des Notebooks.")"""

UPLOAD = f"""## Schritt 5 – Datei ins Repository hochladen / add the file to the repository

1. Öffne <{REPOSITORY}/tree/phase-5-publish/results/phase5/gpu>.
2. *Add file → Upload files*, die heruntergeladene Datei `smolvla_gpu_….json` hineinziehen.
3. Unten *Commit directly to the `phase-5-publish` branch* wählen → *Commit changes*.

Danach rechnet Kortyn mit dieser Datei die präregistrierte Entscheidungsregel (≥ 10 Roboter mit p99 ≤ 150 ms).
Ausführliche Anleitung: [docs/gpu_benchmark_browser.md]({GUIDE}).

*English:* upload the JSON file to `results/phase5/gpu/` on the `phase-5-publish` branch (Add file → Upload \
files → Commit)."""


def _raw_literal(text: str, what: str) -> str:
    """``text`` must survive inside ``r'''...'''`` unchanged."""
    if "'''" in text or text.endswith("\\"):
        raise BuildError(f"{what} cannot be embedded in a raw triple-quoted string")
    return text


def _source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str, hidden: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {"cellView": "form"} if hidden else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _source(text),
    }


def build_notebook(script: str, observations: str) -> dict[str, Any]:
    """The notebook as a JSON document (nbformat 4.5)."""
    files = FILES.format(
        q="'",
        script=_raw_literal(script, "the benchmark script"),
        observations=_raw_literal(observations, "the observations"),
    )
    install = INSTALL.format(
        uv=UV_VERSION, index=TORCH_INDEX, torch=list(TORCH_PACKAGES), packages=list(PACKAGES)
    )
    cells = [
        _markdown(INTRO),
        _code(GPU_CHECK),
        _code(install),
        _code(files, hidden=True),
        _code(RUN),
        _code(DOWNLOAD),
        _markdown(UPLOAD),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"kortyn-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": False},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(notebook: dict[str, Any]) -> str:
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def build_text(script_path: Path = SCRIPT, observations_path: Path = OBSERVATIONS) -> str:
    try:
        script = script_path.read_text(encoding="utf-8")
        observations = json.dumps(
            json.loads(observations_path.read_text(encoding="utf-8")), separators=(",", ":")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read the notebook parts: {exc}") from exc
    return render(build_notebook(script, observations))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build notebooks/gpu_benchmark.ipynb from its parts.")
    parser.add_argument("--out", default=str(NOTEBOOK))
    parser.add_argument("--check", action="store_true", help="only check that the notebook is up to date")
    args = parser.parse_args(argv)
    try:
        text = build_text()
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    if args.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        if current != text:
            print(f"{out} is out of date: run python -m kortyn.notebook_build", file=sys.stderr)
            return 1
        print(f"{out} is up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"written {out} ({len(text) / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
