"""Exception hierarchy for tarash-audio-mixer."""


class AudioMixerException(Exception):
    """Base exception for all audio mixer errors.

    Attributes:
        message: Human-readable error description.
    """

    message: str

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class FFmpegNotFoundError(AudioMixerException):
    """Raised when the FFmpeg binary is not found at the configured path."""

    pass


class InvalidInputError(AudioMixerException):
    """Raised when the input file doesn't exist, isn't readable, or is unsupported."""

    pass


class ProcessingError(AudioMixerException):
    """Raised when FFmpeg processing fails during mixing."""

    pass


class DetectionError(AudioMixerException):
    """Raised when speech detection fails."""

    pass
