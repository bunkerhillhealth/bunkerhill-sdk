"""Class definition for Inference API Client."""

import os
from typing import Any, Dict, Final, List, Optional

import aiohttp
from asgiref.sync import async_to_sync
from retry import retry

from .django_jwt_client import DjangoJWTClient
from .exceptions import (
    FailedToWriteSegmentationError,
    InvalidInferenceAPIClientArgsException,
    SegmentationDownloadError,
)
from .types import Inference


class InferenceAPIClient:
    """Client for interacting with the Bunkerhill Health Inference API."""

    AUTH_PATH: Final[str] = "api/auth/jwt_login/"
    GET_INFERENCE_PATH: Final[str] = "api/models/{model_id}/patients/{patient_mrn}/inferences/"

    _session: aiohttp.ClientSession

    def __init__(
        self,
        username: str,
        private_key_filename: Optional[str] = None,
        private_key_string: Optional[str] = None,
        base_url: str = "https://api.bunkerhillhealth.com/",
    ) -> None:
        """Constructs an InferenceAPIClient.

        Args:
          username: Username for authentication.
          private_key_filename: Path to the private key file for authentication.
          private_key_string: String representation of the private key for authentication.
          base_url: Base URL for the API. Defaults to 'https://api.bunkerhillhealth.com/'.
        Raises:
          InvalidInferenceAPIClientArgsException: If invalid arguments are passed to the constructor.
        """
        self._validate_args(private_key_filename, private_key_string)
        if private_key_filename:
            with open(private_key_filename, "r") as f:
                client_private_key = f.read()
        elif private_key_string:
            client_private_key = private_key_string
        self._django_jwt_client = DjangoJWTClient(
            username=username,
            client_private_key=client_private_key,
            base_url=base_url,
            auth_path=self.AUTH_PATH,
        )

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._session.close()
        self._session = None

    def _validate_args(
        self,
        private_key_filename: Optional[str] = None,
        private_key_string: Optional[str] = None,
    ) -> None:
        if not private_key_filename and not private_key_string:
            raise InvalidInferenceAPIClientArgsException(
                reason="either private_key_filename or private_key_string must be provided"
            )
        if private_key_filename and private_key_string:
            raise InvalidInferenceAPIClientArgsException(
                reason="private_key_filename and private_key_string cannot both be provided"
            )

    async def get_inferences(
        self,
        model_id: str,
        patient_mrn: str,
        segmentation_destination_dirname: str,
    ) -> List[Inference]:
        """Asynchronous function to fetch inferences from the API matching the model_id and patient_mrn, and
        download their segmentations locally.

        Args:
          model_id: Model ID for the inferences.
          patient_mrn: Medical record number (MRN) of the patient.
          segmentation_destination_dirname: Directory name where the segmentations will be downloaded.

        Returns:
          List of Inferences (each in Dict form, loaded from JSON) fetched from the API.

        Raises:
          InferenceAPIRequestFailedError: If a 400- or 500- response is received from the server.
          JSONResponseParseError: If a 200- response is received from the server, but it contains invalid JSON.
          SegmentationDownloadError: If one or more segmentations at the URLs returned by the server fail to download.
          FailedToWriteSegmentationError: If segmentation_destination_dirname is invalid or lacks write permissions.
        """
        resource_path = self.GET_INFERENCE_PATH.format(
            model_id=model_id,
            patient_mrn=patient_mrn,
        )
        return await self._django_jwt_client.get_json(resource_path, self._session)
