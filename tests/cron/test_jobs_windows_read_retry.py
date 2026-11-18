"""Regression tests for transient Windows read contention in utils.read_text_windows_retry and cron/jobs.py."""

import builtins
import errno
import json
import os
from pathlib import Path

import pytest

import utils as utils_mod
from cron import jobs as jobs_mod


def _win_perm_error(winerror: int) -> PermissionError:
    exc = PermissionError(errno.EACCES, "Permission denied", "jobs.json")
    exc.winerror = winerror
    return exc


@pytest.fixture()
def fast_read_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils_mod, "_REPLACE_RETRY_BASE_DELAY_S", 0.001)
    monkeypatch.setattr(utils_mod, "_REPLACE_RETRY_MAX_DELAY_S", 0.001)


@pytest.fixture()
def isolated_cron_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", cron_dir / "output")
    return cron_dir


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_read_text_windows_retry_recovers_from_contended_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_read_retries: None,
    winerror: int,
) -> None:
    target = tmp_path / "jobs.json"
    target.write_text('{"jobs": []}', encoding="utf-8")

    monkeypatch.setattr(utils_mod, "_IS_WINDOWS", True)
    real_open = builtins.open
    attempts = {"count": 0}

    def flaky_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == target:
            attempts["count"] += 1
            if attempts["count"] <= 2:
                raise _win_perm_error(winerror)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky_open)

    assert utils_mod.read_text_windows_retry(target, encoding="utf-8-sig") == '{"jobs": []}'
    assert attempts["count"] == 3


def test_read_text_windows_retry_propagates_permanent_denial_and_load_jobs_wraps_it(
    isolated_cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_read_retries: None,
) -> None:
    jobs_file = isolated_cron_store / "jobs.json"
    jobs_file.write_text('{"jobs": []}', encoding="utf-8")

    monkeypatch.setattr(utils_mod, "_IS_WINDOWS", True)
    real_open = builtins.open
    attempts = {"count": 0}

    def always_denied(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == jobs_file:
            attempts["count"] += 1
            raise _win_perm_error(5)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", always_denied)

    with pytest.raises(PermissionError):
        utils_mod.read_text_windows_retry(jobs_file, encoding="utf-8-sig")
    assert attempts["count"] == 1 + utils_mod._REPLACE_RETRY_ATTEMPTS

    attempts["count"] = 0
    with pytest.raises(RuntimeError, match="Failed to read cron database"):
        jobs_mod.load_jobs()
    assert attempts["count"] == 1 + utils_mod._REPLACE_RETRY_ATTEMPTS


def test_read_text_windows_retry_skips_non_contended_errors_and_non_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_read_retries: None,
) -> None:
    target = tmp_path / "missing.json"
    monkeypatch.setattr(utils_mod, "_IS_WINDOWS", True)
    real_open = builtins.open
    calls = {"count": 0}

    def missing_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == target:
            calls["count"] += 1
            exc = FileNotFoundError(errno.ENOENT, "No such file", str(target))
            exc.winerror = 2
            raise exc
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", missing_open)
    with pytest.raises(FileNotFoundError):
        utils_mod.read_text_windows_retry(target)
    assert calls["count"] == 1

    target.write_text("ok", encoding="utf-8")
    monkeypatch.setattr(utils_mod, "_IS_WINDOWS", False)
    calls["count"] = 0

    def denied_non_windows(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == target:
            calls["count"] += 1
            raise _win_perm_error(5)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied_non_windows)
    with pytest.raises(PermissionError):
        utils_mod.read_text_windows_retry(target)
    assert calls["count"] == 1


def test_load_jobs_recovers_from_transient_winerror_5(
    isolated_cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_read_retries: None,
) -> None:
    jobs_file = isolated_cron_store / "jobs.json"
    payload = {"jobs": [{"id": "job-1", "name": "daily", "prompt": "hi", "enabled": True}]}
    jobs_file.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(utils_mod, "_IS_WINDOWS", True)
    real_open = builtins.open
    attempts = {"count": 0}

    def flaky_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file) == jobs_file:
            attempts["count"] += 1
            if attempts["count"] <= 2:
                raise _win_perm_error(5)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky_open)

    loaded = jobs_mod.load_jobs()
    assert [j["id"] for j in loaded] == ["job-1"]
    assert attempts["count"] == 3


@pytest.mark.platforms("windows")
def test_parse_jobs_file_recovers_from_torn_json_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs_file = tmp_path / "jobs.json"
    reads = iter(['{"jobs": [{"id": "cut', '{"jobs": [{"id": "ok"}]}'])
    sleeps: list[float] = []

    monkeypatch.setattr(jobs_mod, "read_text_windows_retry", lambda *_a, **_kw: next(reads))
    monkeypatch.setattr(jobs_mod.time, "sleep", sleeps.append)

    data, strict_retry = jobs_mod._parse_jobs_file(jobs_file)
    assert data == {"jobs": [{"id": "ok"}]}
    assert strict_retry is False
    assert sleeps == [0.05]
