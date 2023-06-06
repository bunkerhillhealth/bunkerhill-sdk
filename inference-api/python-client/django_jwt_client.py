import json
import jwt
import requests
import traceback

from datetime import datetime
from retry import retry
from typing import Any, Dict, Final, Optional

import pytz

from .exceptions import JSONResponseParseError, InferenceApiRequestFailedError


class DjangoJWTClient:
  CLIENT_JWT_ENCODING_ALGORITHM: Final[str] = 'RS256'
  AUTH_REQUEST_HEADERS: Final[Dict[str, str]] = {
    'Content-Type': 'application/json',
  }

  _django_base_url: str
  _auth_path: str
  _num_failures: int = 0
  _num_failures_allowed: int
  _client_private_key: str

  _access_jwt: Optional[jwt.JWT] = None

  def __init__(
    self,
    client_private_key: str,
    base_url: str,
    auth_path: str,
    num_failures_allowed: int = 3
  ) -> None:
    self._client_private_key = client_private_key
    self._django_base_url = base_url
    self._auth_path = auth_path
    self._num_failures_allowed = num_failures_allowed

  @retry(tries=3, delay=1, backoff=2)
  async def get_json(self, resource_path: str) -> Any:
    await self._ensure_jwt()
    headers = self._create_request_header()
    url = self._django_base_url + resource_path
    
    try:
      response = requests.get(
        url=url,
        headers=headers)
      status = response.status_code
      response.raise_for_status()
    except:
      self._handle_failed_request(url, 'GET', status)

    try:
      response_json = json.loads(response.content)
    except:
      raise JSONResponseParseError()
    else:
      self._num_failures = 0
    return response_json

  async def _ensure_jwt(self) -> None:
    if self._should_refresh_jwt():
      await self._authorize()

  def _should_refresh_jwt(self) -> bool:
    return (not self._access_jwt) or self._is_jwt_expired()

  def _is_jwt_expired(self) -> bool:
    decoded = jwt.decode(self._access_jwt, options={"verify_signature": False})
    exp_timestamp = decoded.get('exp')
    exp = datetime.fromtimestamp(exp_timestamp)
    return datetime.utcnow() > exp

  async def _authorize(self):
    data_to_encode = {
      'iss': 'inference_api_python_client',
      'exp': datetime.utcnow() + timedelta(minutes=30),
    }
    client_jwt = jwt.encode(data_to_encode, self._client_private_key, algorithm=self.CLIENT_JWT_ENCODING_ALGORITHM)
    self._access_jwt = await self._send_authorization_request(client_jwt)

  @retry(tries=3, delay=1, backoff=2)
  async def _send_authorization_request(self, client_jwt: jwt.JWT) -> jwt.JWT:
    url = self._django_base_url + self._auth_path
    response = requests.post(
      url=url,
      headers=self.AUTH_REQUEST_HEADERS,
      data={
        'jwt': client_jwt.decode('utf-8')
      })
    status = response.status_code
    if status >= 400:
      raise InferenceApiRequestFailedError(url, 'POST', status_code)
    response_json = json.loads(response.content)    
    return response_json['token']

  def _handle_failed_request(self, url: str, action: str, status_code: int):
    self._num_failures += 1
    if self._should_reset_access_jwt():
      self._access_jwt = None
    raise InferenceApiRequestFailedError(url, action, status_code)

  def _should_reset_access_jwt(self) -> bool:
    return self._is_jwt_expired() or self._num_failures % self._num_failures_allowed == 1

  def _create_request_header(self) -> Dict[str, str]:
    return {
      'Authorization': f'Bearer {self._access_jwt}',
    }
