"""Data models for caption generation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaptionRequest(BaseModel):
    """Request for generating a caption."""

    prompt: str = Field(description="The video context or prompt for caption generation")
    language: str = Field(default="en", description="Language code for the caption")
    max_length: int | None = Field(default=None, description="Maximum caption length")


class CaptionResponse(BaseModel):
    """Response from caption generation."""

    text: str = Field(description="The generated caption text")
    language: str = Field(description="Language code of the caption")


class CaptionConfig(BaseModel):
    """Configuration for caption generation provider."""

    provider: str = Field(description="Provider name (e.g., 'openai')")
    api_key: str = Field(description="API key for the provider")
    model: str = Field(default="gpt-4o", description="Model identifier")
