"""Exception hierarchy for tarash-silence-remover."""


class SilenceRemoverException(Exception):
    """Base exception for all silence remover errors.

    Attributes:
        message: Human-readable error description.
    """

    message: str

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class FFmpegNotFoundError(SilenceRemoverException):
    """Raised when the FFmpeg binary is not found at the configured path."""

    pass


class InvalidInputError(SilenceRemoverException):
    """Raised when the input file doesn't exist, isn't readable, or is unsupported."""

    pass


class ProcessingError(SilenceRemoverException):
    """Raised when FFmpeg processing fails during cutting or concatenation."""

    pass


class DetectionError(SilenceRemoverException):
    """Raised when silence detection fails."""

    pass
