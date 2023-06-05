from typing import Optional

class InvalidInferenceAPIClientArgsException(Exception):

  def __init__(
    self,
    reason: Optional[str] = None,
  ) -> None:
    message = 'Invalid Inference API Client Arguments'
    if reason:
      message += f': {reason}'
    super().__init__(message)
