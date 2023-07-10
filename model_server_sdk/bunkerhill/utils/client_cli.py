"""CLI for sending requests to inference server.

Example usage:
  python client_cli.py \
    --socket_dirname=/tmp \
    --mounted_data_dirname=/data \
    --study_uuid=fc38b3a8-1eed-4759-ba01-09abf7a935fd
"""

import argparse
import json

from typing import Any, Final, List, Tuple

import grpc

from bunkerhill.proto.inference_pb2 import InferenceRequest
from bunkerhill.proto.inference_pb2 import InferenceResponse
from bunkerhill.proto.inference_pb2_grpc import InferenceProcessorStub

SERVICE_CONFIG_JSON: Final[str] = json.dumps(
  {
    'methodConfig': [
      {
        # To apply retry to all methods, put [{}] in the 'name' field
        'name': [{
          'service': 'InferenceProcessorWorker',
          'method': 'RunInference'
        }],
        'retryPolicy': {
          'maxAttempts': 5,
          'initialBackoff': '0.1s',
          'maxBackoff': '1s',
          'backoffMultiplier': 2,
          'retryableStatusCodes': ['UNAVAILABLE'],
        },
      }
    ]
  }
)
SERVICE_OPTIONS: Final[List[Tuple[str, Any]]] = [
  ('grpc.enable_retries', 1), ('grpc.service_config', SERVICE_CONFIG_JSON)
]
UNIX_DOMAIN_SOCKET_PATH_TEMPLATE = 'unix://%s/model_runner.sock'


def start_grpc_client(socket_dirname: str) -> InferenceProcessorStub:
  channel = grpc.insecure_channel(
    UNIX_DOMAIN_SOCKET_PATH_TEMPLATE % socket_dirname, options=SERVICE_OPTIONS
  )
  # Blocks until channel is ready.
  grpc.channel_ready_future(channel).result()
  stub = InferenceProcessorStub(channel)
  return stub


def main(args: argparse.Namespace) -> None:
  stub = start_grpc_client(args.socket_dirname)
  print('Sending InferenceRequest for study:', args.study_uuid)
  response: InferenceResponse = stub.RunInference(
    InferenceRequest(data_dirname=args.mounted_data_dirname, uuid=args.study_uuid),
    wait_for_ready=True
  )
  print('Received InferenceResponse for study:', args.study_uuid)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--socket_dirname',
                      help='Directory where model runner socket is stored.',
                      required=True,
                      type=str)
  parser.add_argument('--mounted_data_dirname',
                      help=('Mounted directory on Docker image where model runner can read and '
                            'write model inputs and outputs.'),
                      required=True,
                      type=str)
  parser.add_argument('--study_uuid',
                      help=('UUID identifier of study. Inputs and outputs must be encoded as '
                            '{study_uuid}_input.pkl and {study_uuid}_output.pkl respectively.'),
                      required=True,
                      type=str)
  main(parser.parse_args())
