"""``runtime.io_profile`` from YAML must actually steer the I/O profile.

The HDD clamp tells users to set ``runtime.io_profile: hdd`` in the config,
so that knob has to win over sysfs auto-detection without an env var present.
Precedence under test: explicit override / ``$BALALAIKA_IO_PROFILE`` > YAML
``runtime.io_profile`` > sysfs auto-detect.
"""
from __future__ import annotations

import textwrap

import pytest

from src.utils import io_profile
from src.utils import runtime_env


@pytest.fixture(autouse=True)
def _clear_caches():
    io_profile.resolve_io_profile.cache_clear()
    runtime_env._runtime_cfg_cached.cache_clear()
    yield
    io_profile.resolve_io_profile.cache_clear()
    runtime_env._runtime_cfg_cached.cache_clear()


def _write_config(tmp_path, profile):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""\
            runtime:
              io_profile: {profile}
            """
        ),
        encoding="utf-8",
    )
    return cfg


def _detector_must_not_run(monkeypatch):
    """Make sysfs auto-detection blow up if anything reaches it."""

    def _boom(_path):
        raise AssertionError("sysfs auto-detection consulted despite explicit profile")

    monkeypatch.setattr(io_profile, "is_rotational", _boom)


def test_yaml_hdd_wins_without_sysfs(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, "hdd")
    monkeypatch.delenv(io_profile.IO_PROFILE_ENV, raising=False)
    monkeypatch.setenv("BALALAIKA_CONFIG_PATH", str(cfg))
    _detector_must_not_run(monkeypatch)

    assert io_profile.resolve_io_profile(str(tmp_path)) == "hdd"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, "hdd")
    monkeypatch.setenv("BALALAIKA_CONFIG_PATH", str(cfg))
    monkeypatch.setenv(io_profile.IO_PROFILE_ENV, "ssd")
    _detector_must_not_run(monkeypatch)

    assert io_profile.resolve_io_profile(str(tmp_path)) == "ssd"


def test_neither_set_falls_back_to_autodetect(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, "auto")
    monkeypatch.delenv(io_profile.IO_PROFILE_ENV, raising=False)
    monkeypatch.setenv("BALALAIKA_CONFIG_PATH", str(cfg))

    sentinel_calls = []

    def _fake_rotational(path):
        sentinel_calls.append(path)
        return True  # pretend the device is rotational

    monkeypatch.setattr(io_profile, "is_rotational", _fake_rotational)

    assert io_profile.resolve_io_profile(str(tmp_path)) == "hdd"
    assert sentinel_calls == [str(tmp_path)]


def test_clamp_loader_workers_honors_yaml_hdd(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, "hdd")
    monkeypatch.delenv(io_profile.IO_PROFILE_ENV, raising=False)
    monkeypatch.setenv("BALALAIKA_CONFIG_PATH", str(cfg))
    _detector_must_not_run(monkeypatch)

    files = [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]
    assert (
        io_profile.clamp_loader_workers(16, files)
        == io_profile.HDD_MAX_LOADER_WORKERS
    )
