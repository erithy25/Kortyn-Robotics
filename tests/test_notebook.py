"""The GPU notebook (notebooks/gpu_benchmark.ipynb): built from its parts, and its GPU check without a GPU."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from kortyn import notebook_build as build

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def notebook() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(build.NOTEBOOK.read_text(encoding="utf-8"))
    return data


def _code_cells(notebook: dict[str, Any]) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_committed_notebook_is_the_current_build() -> None:
    assert build.NOTEBOOK.read_text(encoding="utf-8") == build.build_text(), (
        "notebooks/gpu_benchmark.ipynb is out of date: python -m kortyn.notebook_build"
    )


def test_notebook_structure(notebook: dict[str, Any]) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    kinds = [cell["cell_type"] for cell in notebook["cells"]]
    assert kinds == ["markdown", "code", "code", "code", "code", "code", "markdown"]
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(set(ids)) == len(ids)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_every_code_cell_is_valid_python(notebook: dict[str, Any]) -> None:
    for source in _code_cells(notebook):
        ast.parse(source)


def _assigned_string(source: str, name: str) -> str:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            value = node.value
            assert isinstance(value, ast.Constant)
            assert isinstance(value.value, str)
            return value.value
    raise AssertionError(f"{name} is not assigned in the cell")


def test_embedded_script_and_observations_are_verbatim(notebook: dict[str, Any]) -> None:
    files_cell = _code_cells(notebook)[2]
    assert _assigned_string(files_cell, "SCRIPT") == build.SCRIPT.read_text(encoding="utf-8")
    observations = json.loads(_assigned_string(files_cell, "OBSERVATIONS"))
    assert observations == json.loads(build.OBSERVATIONS.read_text(encoding="utf-8"))


def test_pinned_versions_match_the_lock_file() -> None:
    """The notebook installs the versions of the CPU experiments (uv.lock)."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"].split("+")[0])
    for requirement in (*build.TORCH_PACKAGES, *build.PACKAGES):
        name, _, version = requirement.partition("==")
        name = name.split("[")[0]
        assert version in versions[name], f"{name}=={version} not in uv.lock ({versions.get(name)})"


def test_raw_literal_guard() -> None:
    with pytest.raises(build.BuildError):
        build._raw_literal("a ''' b", "x")
    with pytest.raises(build.BuildError):
        build._raw_literal("ends with \\", "x")
    assert build._raw_literal('fine """ text', "x") == 'fine """ text'


def test_check_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stale = tmp_path / "nb.ipynb"
    stale.write_text("{}", encoding="utf-8")
    assert build.main(["--check", "--out", str(stale)]) == 1
    assert build.main(["--out", str(stale)]) == 0
    assert build.main(["--check", "--out", str(stale)]) == 0
    capsys.readouterr()


# --------------------------------------------------------------------------------------------- GPU check
def _run_gpu_check(
    notebook: dict[str, Any], monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> dict[str, Any]:
    for key in ("COLAB_RELEASE_TAG", "KAGGLE_KERNEL_RUN_TYPE", "KORTYN_BENCHMARK_ALLOW_CPU"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    exec(compile(_code_cells(notebook)[0], "gpu_check", "exec"), namespace)
    return namespace


def _class_named(name: str, error: BaseException) -> bool:
    return type(error).__name__ == name


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "keine NVIDIA-GPU"),
        ({"COLAB_RELEASE_TAG": "release"}, "Laufzeittyp ändern"),
        ({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, "GPU T4 x2"),
    ],
)
def test_gpu_check_stops_without_gpu(
    notebook: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env: dict[str, str],
    expected: str,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError) as caught:
        _run_gpu_check(notebook, monkeypatch, env)
    assert _class_named("KeineGPU", caught.value)
    assert "Keine GPU aktiv" in str(caught.value)
    assert expected in capsys.readouterr().out


def test_gpu_check_finds_a_gpu(notebook: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[0] == "/usr/bin/nvidia-smi"
        return subprocess.CompletedProcess(command, 0, stdout="Tesla T4, 15360 MiB, 550.54.15\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    namespace = _run_gpu_check(notebook, monkeypatch, {"COLAB_RELEASE_TAG": "release"})
    assert namespace["GPUS"] == ["Tesla T4, 15360 MiB, 550.54.15"]
    assert namespace["PLATFORM"] == "colab"
    assert str(namespace["WORK_DIR"]) == "/content"


def test_gpu_check_treats_a_failing_nvidia_smi_as_no_gpu(
    notebook: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def failing(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="NVIDIA-SMI has failed")

    monkeypatch.setattr(subprocess, "run", failing)
    with pytest.raises(RuntimeError):
        _run_gpu_check(notebook, monkeypatch, {})
    capsys.readouterr()


def test_gpu_check_test_switch_allows_the_cpu(
    notebook: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    namespace = _run_gpu_check(notebook, monkeypatch, {"KORTYN_BENCHMARK_ALLOW_CPU": "1"})
    assert namespace["GPUS"] == []
    assert namespace["ALLOW_CPU"] is True
    assert "Testmodus" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------- installation
def _run_install(
    notebook: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    venv: Path,
    python_is_312: bool,
) -> list[list[str]]:
    """Run the install cell with every subprocess faked; return the commands it ran."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:3] == ["-c", "import sys; print(sys.version_info[:2] == (3, 12))"]:
            return subprocess.CompletedProcess(command, 0, stdout=f"{python_is_312}\n", stderr="")
        if command[1:2] == ["-c"]:
            return subprocess.CompletedProcess(command, 0, stdout="2.11.0 12.6 True\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("KORTYN_BENCHMARK_VENV", str(venv))
    namespace: dict[str, Any] = {"__name__": "__notebook__", "ALLOW_CPU": False}
    exec(compile(_code_cells(notebook)[1], "install", "exec"), namespace)
    return commands


def _creates_venv(commands: list[list[str]]) -> bool:
    return any("venv" in command[3:5] for command in commands)


def test_install_creates_the_environment(
    notebook: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands = _run_install(notebook, monkeypatch, tmp_path / "venv", python_is_312=True)
    venv_commands = [command for command in commands if "venv" in command[3:5]]
    assert len(venv_commands) == 1
    assert "--clear" in venv_commands[0]
    assert sum("pip" in command[3:5] and "install" in command for command in commands) == 2
    capsys.readouterr()


def test_install_reuses_the_environment_of_an_earlier_run(
    notebook: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second "Run all" in the same session must not fail because the environment exists."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    commands = _run_install(notebook, monkeypatch, venv, python_is_312=True)
    assert not _creates_venv(commands)
    assert "schon vorhanden" in capsys.readouterr().out
    commands = _run_install(notebook, monkeypatch, venv, python_is_312=False)
    assert _creates_venv(commands), "an environment with another Python version is replaced"
    capsys.readouterr()


# --------------------------------------------------------------------------------------------- download
def test_download_without_a_result_file_explains_why(notebook: dict[str, Any]) -> None:
    namespace: dict[str, Any] = {"__name__": "__notebook__", "RESULT": None}
    with pytest.raises(RuntimeError, match="keine Ergebnisdatei"):
        exec(compile(_code_cells(notebook)[4], "download", "exec"), namespace)
    with pytest.raises(RuntimeError, match="keine Ergebnisdatei"):
        exec(compile(_code_cells(notebook)[4], "download", "exec"), {"__name__": "__notebook__"})


def test_download_outside_colab_and_kaggle_names_the_file(
    notebook: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = tmp_path / "smolvla_gpu_teslat4_other_20261028-0930.json"
    result.write_text("{}", encoding="utf-8")
    namespace: dict[str, Any] = {
        "__name__": "__notebook__",
        "RESULT": result,
        "ON_COLAB": False,
        "ON_KAGGLE": False,
    }
    exec(compile(_code_cells(notebook)[4], "download", "exec"), namespace)
    out = capsys.readouterr().out
    assert str(result) in out
    assert "Arbeitsverzeichnis" in out
