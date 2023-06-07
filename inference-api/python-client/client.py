import requests

from asgiref.sync import async_to_sync
from retry import retry
from typing import Final, Optional

from .exceptions import (
  FailedToWriteSegmentationError,
  InvalidInferenceAPIClientArgsException,
  SegmentationDownloadError,
)
from .types import Inference
from .django_jwt_client import DjangoJWTClient

class InferenceAPIClient:
  AUTH_PATH: Final[str] = '/api/auth/jwt_login'
  GET_INFERENCE_PATH: Final[str] = 'models/{model_id}/patients/{patient_mrn}/inferences'

  def __init__(
    self,
    private_key_filename: Optional[str] = None,
    private_key_string: Optional[str] = None,
    base_url: str = 'https://api.bunkerhillhealth.com/',
  ) -> None:
    self._validate_args(private_key_filename, private_key_string)
    if private_key_filename:
      with open(private_key_filename, 'r') as f:
        client_private_key = f.read()
    else private_key_string:
      client_private_key = private_key_string
    self._django_jwt_client = DjangoJWTClient(
      client_private_key=client_private_key,
      base_url=base_url,
      auth_path=self.AUTH_PATH,
    )

  def _validate_args(
    self,
    private_key_filename: Optional[str] = None,
    private_key_string: Optional[str] = None,
  ) -> None:
    if not private_key_filename and not private_key_string:
      raise InvalidInferenceAPIClientArgsException(
        reason='either private_key_filename or private_key_string must be provided')
    if private_key_filename and private_key_string:
      raise InvalidInferenceAPIClientArgsException(
        reason='private_key_filename and private_key_string cannot both be provided')

  async def get_inference_async(
    self,
    model_id: str,
    patient_mrn: str,
    segmentation_destination_dirname: str,
  ) -> List[Inference]:
    resource_path = self.GET_INFERENCE_PATH.format(
      model_id=model_id,
      study_instance_uid=study_instance_uid,
      series_instance_uid=series_instance_uid,
    )
    response_json = await self._django_jwt_client.get_json(resource_path)
    inferences: List[Inference] = []
    for inference in response_json:
      self._download_segmentations(
        presigned_urls=response_json['segmentation_presigned_urls'],
        destination_dirname=segmentation_destination_dirname,
      )
      inferences.append(
        Inference(
          model_id=model_id,
          patient_mrn=patient_mrn,
          segmentation_presigned_urls=response_json['segmentation_presigned_urls'],
      ))
    return inferences

  def get_inference(
    self,
    model_id: str,
    patient_mrn: str,
    segmentation_destination_dirname: str,
  ) -> Inference:
    return async_to_sync(self.get_inference_async)(
      model_id=model_id,
      patient_mrn=patient_mrn,
      segmentation_destination_dirname=segmentation_destination_dirname,
    )

  def _download_segmentations(
    self,
    presigned_urls: str,
    destination_dirname: str,
  ) -> None:
    for presigned_url in presigned_urls:
      basename = 'TODO' # TODO: Compute basename of downloaded presigned URL based on returned data.
      destination_filename = os.path.join(destination_dirname, destination_basename)
      try:
        response = requests.get(presigned_url)
        response.raise_for_status()
        content = response.content
      except:
        raise SegmentationDownloadError(presigned_url)
      try:
        with open(destination_filename, 'wb') as f:
          f.write(content)
      except:
          raise FailedToWriteSegmentationError(destination_filename)
