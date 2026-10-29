"""Keep the suite offline and out of the user's cache.

Every test gets its own allelign cache directory, and the module-level
download hook refuses network access; tests that exercise downloads pass a
fake ``opener`` or patch the chain loader instead.
"""
import pytest

from allelign import download


def _no_network(url, *args, **kwargs):
    raise OSError(f"network access in a test: {url}")


@pytest.fixture(autouse=True)
def offline_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(download.CACHE_DIR_ENV, str(tmp_path / "allelign-cache"))
    monkeypatch.setattr(download, "_open", _no_network)
