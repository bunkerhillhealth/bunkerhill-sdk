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


class SegmentationDownloadError(Exception):

  def __init__(
    self,
    presigned_url: str,
    destination_filename: str,
    message: str = 'Failed to download segmentation from {presigned_url}',
  ) -> None:
    super().__init__(message.format(presigned_url=presigned_url))


class FailedToWriteSegmentationError(Exception):

  def __init__(
    self,
    destination_filename: str,
    message: str = 'Failed to write downloaded segmentation file to {destination_filename}',
  ) -> None:
    super().__init__(message.format(destination_filename=destination_filename))


class JSONResponseParseError(Exception):

  def __init__(
    self,
    message: str = 'Error parsing JSON response from Inference API server',
  ) -> None:
    super().__init__(message)


class InferenceAPIRequestFailedError(Exception):

  def __init__(
    self,
    url: str,
    action: str,
    status_code: int,
    message: str = 'Error connecting to Inference API: {action} {url} returned a status code of {status_code}',
  ) -> None:
    super().__init__(message.format(
      url=url,
      action=action,
      status_code=status_code,
    ))
