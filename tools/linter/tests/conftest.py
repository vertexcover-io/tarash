"""Shared test fixtures for tarash-lint tests."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_gateway(tmp_path: Path) -> Path:
    """Create a minimal fake tarash-gateway directory structure."""
    gw = tmp_path / "packages" / "tarash-gateway"
    providers_dir = gw / "src" / "tarash" / "tarash_gateway" / "providers"
    providers_dir.mkdir(parents=True)

    # Create a valid video provider
    (providers_dir / "fakevideo.py").write_text(
        '''"""Fake video provider."""

class FakevideoProviderHandler:
    def _get_client(self, config, client_type): ...
    def _handle_error(self, config, request, request_id, ex): ...
    def _convert_request(self, config, request): ...
    def _convert_response(self, config, request, request_id, response): ...
    def _validate_params(self, config, request): ...
    async def generate_video_async(self, config, request, on_progress=None): ...
    def generate_video(self, config, request, on_progress=None): ...
'''
    )

    # Create a valid audio provider
    (providers_dir / "fakeaudio.py").write_text(
        '''"""Fake audio provider."""

class FakeaudioProviderHandler:
    def _get_client(self, config, client_type): ...
    def _handle_error(self, config, request, request_id, ex): ...
    async def generate_tts_async(self, config, request, on_progress=None): ...
    def generate_tts(self, config, request, on_progress=None): ...
'''
    )

    # Create __init__.py and field_mappers.py (should be excluded)
    (providers_dir / "__init__.py").write_text('"""Provider exports."""\n')
    (providers_dir / "field_mappers.py").write_text('"""Field mappers."""\n')

    return tmp_path
