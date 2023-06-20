"""Client for connecting to Bunkerhill's Django server, which serves the Inference API"""

import os
import json
import traceback

from datetime import datetime, timedelta
from typing import Any, Dict, Final, Optional

import aiohttp
import jwt
import pytz

from retry import retry

from .exceptions import JSONResponseParseError, InferenceAPIRequestFailedError


class DjangoJWTClient:
  """Client for Bunkerhill's Django server, using JWT authentication"""

  CLIENT_JWT_ENCODING_ALGORITHM: Final[str] = 'RS256'
  AUTH_REQUEST_HEADERS: Final[Dict[str, str]] = {
    'Content-Type': 'application/json',
  }

  _django_base_url: str
  _auth_path: str
  _num_failures: int = 0
  _num_failures_allowed: int
  _username: str
  _client_private_key: str

  _access_jwt: Optional[str] = None

  def __init__(
    self,
    username: str,
    client_private_key: str,
    base_url: str,
    auth_path: str,
    num_failures_allowed: int = 3
  ) -> None:
    """Constructs a DjangoJWTClient.

    Args:
      username (str): Username for authentication.
      client_private_key (str): Private key string for authentication.
      base_url (str): Base URL for the API.
      auth_path (str): Path to the authentication endpoint for the API.
      num_failures_allowed (int): Number of failed requests before attempting to refresh the auth JWT. Default 3.
    
    Raises:
      InferenceAPIRequestFailedError: If a 400- or 500- response is received from the server.
      JSONResponseParseError: If a 200- response is received from the server, but it contains invalid JSON.
    """

    self._username = username
    self._client_private_key = client_private_key
    self._django_base_url = base_url
    self._auth_path = auth_path
    self._num_failures_allowed = num_failures_allowed

  @retry(tries=3, delay=1, backoff=2)
  async def get_json(self, resource_path: str) -> Any:
    """Queries an endpoint and returns the JSON that the server responds with.

    Args:
      resource_path(str): Path to the endpoint to query.

    Returns:
      A JSON object parsed from the server's response.
    """


    await self._ensure_jwt()
    headers = self._create_request_header()
    url = os.path.join(self._django_base_url, resource_path)

    async with aiohttp.ClientSession() as session:
      async with session.get(url, headers=headers) as response:
        return await self._parse_response_json(url, response, action='GET')

  async def _parse_response_json(self, url: str, response: aiohttp.ClientResponse, action: str) -> Any:
    if response.status >= 400:
      await self._handle_failed_request(url, action, response)
    try:
      response_json = await response.json()
      self._num_failures = 0
      return response_json
    except:
      raise JSONResponseParseError()

  async def _ensure_jwt(self) -> None:
    if self._should_refresh_jwt():
      await self._authorize()

  def _should_refresh_jwt(self) -> bool:
    return (not self._access_jwt) or self._is_jwt_expired_or_invalid()

  def _is_jwt_expired_or_invalid(self) -> bool:
    try:
      decoded = jwt.decode(self._access_jwt, options={"verify_signature": False})
      exp_timestamp = decoded.get('exp')
      exp = datetime.fromtimestamp(exp_timestamp)
      return datetime.utcnow() > exp
    except:
      return True

  async def _authorize(self):
    data_to_encode = {
      'iss': 'inference_api_python_client',
      'exp': datetime.utcnow() + timedelta(minutes=30),
      'username': self._username,
    }
    client_jwt = jwt.encode(data_to_encode, self._client_private_key, algorithm=self.CLIENT_JWT_ENCODING_ALGORITHM)
    self._access_jwt = await self._send_authorization_request(client_jwt)

  @retry(tries=3, delay=1, backoff=2)
  async def _send_authorization_request(self, client_jwt: str) -> str:
    url = os.path.join(self._django_base_url, self._auth_path)

    async with aiohttp.ClientSession() as session:
      async with session.post(
        url,
        headers=self.AUTH_REQUEST_HEADERS,
        json={'jwt': client_jwt},
      ) as response:
        response_json = await self._parse_response_json(url, response, action='POST')
        return response_json['token']

  async def _handle_failed_request(self, url: str, action: str, response: aiohttp.ClientResponse):
    self._num_failures += 1
    if self._should_reset_access_jwt():
      self._access_jwt = None
    try:
      error = await response.json()['detail']
    except:
      error = await response.text()
    raise InferenceAPIRequestFailedError(url, action, response, error)

  def _should_reset_access_jwt(self) -> bool:
    return self._is_jwt_expired_or_invalid() or self._num_failures % self._num_failures_allowed == 1

  def _create_request_header(self) -> Dict[str, str]:
    return {
      'Authorization': f'Bearer {self._access_jwt}',
    }
