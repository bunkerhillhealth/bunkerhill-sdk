"""Exceptions raised in the Inference API Client package"""

from typing import Optional

import requests


class InvalidInferenceAPIClientArgsException(Exception):
  """Exception raised when invalid arguments are passed to the InferenceAPIClient constructor"""

  def __init__(
    self,
    reason: Optional[str] = None,
  ) -> None:
    message = 'Invalid Inference API Client Arguments'
    if reason:
      message += f': {reason}'
    super().__init__(message)


class SegmentationDownloadError(Exception):
  """Exception raised segmentation fails to download from a given presigned URL"""

  def __init__(
    self,
    presigned_url: str,
    destination_filename: str,
    message: str = 'Failed to download segmentation from {presigned_url}',
  ) -> None:
    super().__init__(message.format(presigned_url=presigned_url))


class FailedToWriteSegmentationError(Exception):
  """Exception raised when a downloaded segmentation fails to write to disk"""

  def __init__(
    self,
    destination_filename: str,
    message: str = 'Failed to write downloaded segmentation file to {destination_filename}',
  ) -> None:
    super().__init__(message.format(destination_filename=destination_filename))


class JSONResponseParseError(Exception):
  """Exception raised when a response from the server cannot be parsed into JSON"""


  def __init__(
    self,
    message: str = 'Error parsing JSON response from Inference API server',
  ) -> None:
    super().__init__(message)


class InferenceAPIRequestFailedError(Exception):
  """Exception raised when a request to the Inference API server returns a 400- or 500- status response."""

  def __init__(
    self,
    url: str,
    action: str,
    response: requests.Response,
    message: str = 'Inference API request failed: {action} {url} returned a status code of {status_code} with error: {error_message}',
  ) -> None:
    try:
      error_message = response.json()['detail']
    except:
      error_message = response.text
    super().__init__(message.format(
      url=url,
      action=action,
      status_code=response.status_code,
      error_message=error_message,
    ))
