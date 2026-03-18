"""Tests for provider discovery via AST parsing."""

from pathlib import Path

from tarash_linter.discovery import discover_providers


def test_discover_finds_video_provider(tmp_gateway: Path):
    """Discovery finds the fakevideo provider."""
    providers = discover_providers(tmp_gateway)
    names = {p.name for p in providers}
    assert "fakevideo" in names


def test_discover_finds_audio_provider(tmp_gateway: Path):
    """Discovery finds the fakeaudio provider."""
    providers = discover_providers(tmp_gateway)
    names = {p.name for p in providers}
    assert "fakeaudio" in names


def test_discover_excludes_init_and_field_mappers(tmp_gateway: Path):
    """Discovery excludes __init__.py and field_mappers.py."""
    providers = discover_providers(tmp_gateway)
    names = {p.name for p in providers}
    assert "__init__" not in names
    assert "field_mappers" not in names


def test_discover_extracts_methods(tmp_gateway: Path):
    """Discovery extracts all method names from the handler class."""
    providers = discover_providers(tmp_gateway)
    video = next(p for p in providers if p.name == "fakevideo")
    assert "_get_client" in video.methods
    assert "_handle_error" in video.methods
    assert "_convert_request" in video.methods
    assert "generate_video" in video.methods
    assert "generate_video_async" in video.methods


def test_discover_extracts_class_info(tmp_gateway: Path):
    """Discovery captures class name and line number."""
    providers = discover_providers(tmp_gateway)
    video = next(p for p in providers if p.name == "fakevideo")
    assert video.class_name == "FakevideoProviderHandler"
    assert video.class_line > 0
    assert "providers/fakevideo.py" in video.file


def test_discover_empty_file(tmp_gateway: Path):
    """Discovery handles files with no ProviderHandler class."""
    providers_dir = (
        tmp_gateway
        / "packages"
        / "tarash-gateway"
        / "src"
        / "tarash"
        / "tarash_gateway"
        / "providers"
    )
    (providers_dir / "empty.py").write_text(
        '"""Empty module."""\n\nclass NotAHandler:\n    pass\n'
    )
    providers = discover_providers(tmp_gateway)
    names = {p.name for p in providers}
    assert "empty" not in names
