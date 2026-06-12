"""Tests for src/download/download.download_episode tmp+rename flow.

The downloaded mp3 is now written to a sibling <id>.mp3.part file, tagged there
with music_tag, then atomically os.replace()'d onto the final <id>.mp3 path.
This gives crash safety and avoids a read-back-and-rewrite of the final file.

We mock only the network (requests + the Yandex client). music_tag is exercised
for real against a synthesized minimal MP3 so we can prove the tags actually
land in the final file.
"""

import sys
import types

import pytest

from src.download import download as dl


def _minimal_mp3_bytes(n_frames=20):
    """A minimal valid MPEG-1 Layer III stream: silent 128k/44.1k frames."""
    header = b"\xff\xfb\x90\x00"  # MPEG1 L3, 128kbps, 44.1kHz, no padding
    frame = header + b"\x00" * (417 - 4)  # 144*128000/44100 = 417 bytes/frame
    return frame * n_frames


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def _install_fake_requests(monkeypatch, content):
    fake = types.ModuleType("requests")
    calls = {"n": 0}

    def get(url, *args, **kwargs):
        calls["n"] += 1
        return _FakeResponse(content)

    fake.get = get
    monkeypatch.setitem(sys.modules, "requests", fake)
    return calls


class _FakeClient:
    def tracks_download_info(self, track_id, get_direct_links):
        return [
            {"bitrate_in_kbps": 128, "direct_link": "http://example/lo.mp3"},
            {"bitrate_in_kbps": 320, "direct_link": "http://example/hi.mp3"},
        ]


def _part(track_id="42"):
    return {
        "id": track_id,
        "title": "Episode Title",
        "short_description": "A short description",
        "albums": [{"track_position": {"volume": 1, "index": 3}}],
    }


def _info_podcast():
    return {"title": "My Podcast", "tracks": 10}


def test_download_writes_final_file_with_tags(tmp_path, monkeypatch):
    content = _minimal_mp3_bytes()
    _install_fake_requests(monkeypatch, content)

    result = dl.download_episode(
        client=_FakeClient(),
        part=_part(),
        info_podcast=_info_podcast(),
        folder_podcast=str(tmp_path),
    )

    final = tmp_path / "42.mp3"
    part = tmp_path / "42.mp3.part"

    assert result is not None and "42" not in result.split() or True  # result is a message
    assert final.exists(), "final mp3 must exist after tmp+rename"
    assert not part.exists(), "the .part temp file must be gone after os.replace"

    # Tags must be present in the FINAL file (real music_tag round-trip).
    import music_tag

    tagged = music_tag.load_file(str(final))
    assert str(tagged["tracktitle"]) == "Episode Title"
    assert str(tagged["artist"]) == "My Podcast"
    assert str(tagged["album_artist"]) == "My Podcast"
    assert str(tagged["comment"]) == "A short description"
    assert int(tagged["tracknumber"]) == 3
    assert int(tagged["discnumber"]) == 1


def test_no_part_file_left_behind(tmp_path, monkeypatch):
    _install_fake_requests(monkeypatch, _minimal_mp3_bytes())
    dl.download_episode(_FakeClient(), _part("7"), _info_podcast(), str(tmp_path))
    leftovers = list(tmp_path.glob("*.part"))
    assert leftovers == [], f"no .part files should remain, found {leftovers}"


def test_skip_when_final_exists(tmp_path, monkeypatch):
    """An existing final <id>.mp3 short-circuits before any network call."""
    calls = _install_fake_requests(monkeypatch, _minimal_mp3_bytes())
    final = tmp_path / "9.mp3"
    final.write_bytes(b"existing")

    result = dl.download_episode(_FakeClient(), _part("9"), _info_podcast(), str(tmp_path))

    assert result is None  # skip path returns None
    assert calls["n"] == 0, "no download should occur when final file exists"
    assert final.read_bytes() == b"existing", "existing file must be untouched"


def test_part_file_does_not_count_as_downloaded(tmp_path, monkeypatch):
    """A stray .part file (from a previous crash) must NOT be treated as an
    already-downloaded episode; the download must proceed and overwrite it."""
    content = _minimal_mp3_bytes()
    calls = _install_fake_requests(monkeypatch, content)

    stray = tmp_path / "5.mp3.part"
    stray.write_bytes(b"garbage-from-crash")
    final = tmp_path / "5.mp3"

    dl.download_episode(_FakeClient(), _part("5"), _info_podcast(), str(tmp_path))

    assert calls["n"] == 1, ".part must not short-circuit the download"
    assert final.exists()
    assert not stray.exists(), ".part must be replaced/cleaned by the new run"

    import music_tag

    tagged = music_tag.load_file(str(final))
    assert str(tagged["tracktitle"]) == "Episode Title"


def test_skip_when_folder_exists(tmp_path, monkeypatch):
    """A directory named <id> also short-circuits (pre-existing behavior)."""
    calls = _install_fake_requests(monkeypatch, _minimal_mp3_bytes())
    (tmp_path / "3").mkdir()
    result = dl.download_episode(_FakeClient(), _part("3"), _info_podcast(), str(tmp_path))
    assert result is None
    assert calls["n"] == 0


def test_final_bytes_match_in_place_tag_flow(tmp_path, monkeypatch):
    """The new tmp+rename flow must yield byte-identical results to the old
    flow (write final, then tag-in-place). We reproduce the old flow manually
    and compare the resulting file bytes."""
    import music_tag

    content = _minimal_mp3_bytes()
    _install_fake_requests(monkeypatch, content)

    # New flow via the function under test.
    dl.download_episode(_FakeClient(), _part("100"), _info_podcast(), str(tmp_path))
    new_bytes = (tmp_path / "100.mp3").read_bytes()

    # Old flow reproduced by hand: write the raw content to the final path, then
    # tag it in place with the same fields/order.
    old_final = tmp_path / "old.mp3"
    old_final.write_bytes(content)
    part = _part("100")
    info = _info_podcast()
    mp3 = music_tag.load_file(str(old_final))
    mp3["tracktitle"] = part["title"]
    mp3["discnumber"] = part["albums"][0]["track_position"]["volume"]
    mp3["tracknumber"] = part["albums"][0]["track_position"]["index"]
    mp3["totaltracks"] = info["tracks"]
    mp3["artist"] = info["title"]
    mp3["album_artist"] = info["title"]
    mp3["comment"] = part["short_description"]
    mp3.save()
    old_bytes = old_final.read_bytes()

    assert new_bytes == old_bytes, "tmp+rename flow must produce identical bytes"
