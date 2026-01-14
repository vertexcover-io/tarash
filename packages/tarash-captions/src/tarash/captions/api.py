"""Caption generation API."""

from __future__ import annotations

from tarash.captions.models import CaptionConfig, CaptionRequest, CaptionResponse


def generate_caption(request: CaptionRequest, config: CaptionConfig) -> CaptionResponse:
    """Generate a caption for the given request.

    Args:
        request: The caption generation request
        config: Provider configuration

    Returns:
        CaptionResponse with the generated caption
    """
    # Placeholder implementation
    return CaptionResponse(
        text=f"Caption for: {request.prompt}",
        language=request.language,
    )
