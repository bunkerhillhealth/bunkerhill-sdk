"""Class definition for model inference server."""

import os
import pickle
import traceback

from concurrent import futures
from typing import Any, Dict, Optional

import grpc

from bunkerhill import shared_file_utils
from bunkerhill import grpc_socket
from bunkerhill.bunkerhill_types import Tag
from bunkerhill.proto.inference_pb2 import InferenceRequest
from bunkerhill.proto.inference_pb2 import InferenceResponse
from bunkerhill.proto import inference_pb2_grpc
from bunkerhill.proto.inference_pb2_grpc import InferenceProcessorServicer


class ModelRunner(InferenceProcessorServicer):
  """Server that triggers model inference upon request.

    Attributes:
      _model_release: The model on which to run inference.
      _server: The gRPC server to which the inference processor connects.
  """
  _SOCKET_DIRNAME = '/data'
  _UDS_ADDRESS = grpc_socket.UNIX_DOMAIN_SOCKET_PATH_TEMPLATE % _SOCKET_DIRNAME
  _model_release: Any
  _server: grpc.Server

  def __init__(self, model_release: Any, max_workers: Optional[int] = 2) -> None:
    """Constructs a ModelRunner object.

    Args:
      model_release: The model_release to use. model_release must have an
        `inference()` method.
      max_workers: Numbers of threads to execute calls asynchronously. If None,
        max_workers set changed to min(32, os.cpu_count() + 4). This default
        value preserves at least 5 workers for I/O bound tasks. See:
        https://docs.python.org/3.9/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
    """
    self._model_release = model_release
    self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    

  def RunInference(self, request: InferenceRequest,
                   context: grpc.ServicerContext) -> InferenceResponse:
    """Runs inference upon receiving an InferenceRequest.

    Args:
      request: The ProcessRequest with RPC arguments.
      context: The gRPC context object passed to method implementations.

    Returns:
      The ProcessResponse to send to the client.
    """
    try:
      model_arguments = self._load_model_arguments(request.data_dirname, request.uuid)
    except FileNotFoundError as e:
      context.set_details(f'Failed to load model arguments. Error={e}.')
      context.set_code(grpc.StatusCode.NOT_FOUND)
      return InferenceResponse()

    try:
      outputs = self._process_model_arguments(model_arguments)
    # Broad exception handling for previous unseen error types.
    except Exception as e:
      print(traceback.format_exc())
      context.set_details(f'Unable to run inference on uuid={request.uuid}. Error={e}.')
      context.set_code(grpc.StatusCode.UNKNOWN)
      return InferenceResponse()

    self._dump_model_outputs(request.data_dirname, request.uuid, outputs)
    return InferenceResponse()

  def start_run_loop(self) -> None:
    """Starts the gRPC server."""
    inference_pb2_grpc.add_InferenceProcessorServicer_to_server(self, self._server)
    self._server.add_insecure_port(ModelRunner._UDS_ADDRESS)
    print(f'Server listening on: {ModelRunner._UDS_ADDRESS}.')
    self._server.start()
    self._server.wait_for_termination()

  def _process_model_arguments(self, model_arguments: Dict[Tag, Any]) -> Dict[Tag, Any]:
    outputs = self._run_inference(model_arguments)
    print(f'Received model outputs: {outputs}')
    return outputs

  def _run_inference(self, model_arguments: Dict[Tag, Any]) -> Dict[Tag, Any]:
    """Runs model inference. This method is exposed for mocking in tests."""
    return self._model_release.inference(**model_arguments)

  def _dump_model_outputs(self, data_dirname: str, uuid: str,
                          outputs: Dict[Tag, Any]) -> None:
    model_outputs_filename = shared_file_utils.get_model_outputs_filename(data_dirname,
                                                                          uuid)
    try:
      with open(model_outputs_filename, 'wb') as f:
        pickle.dump(outputs, f)
    except OSError as e:
      print(f'Failed to dump to filename={model_outputs_filename} with error={e}.')
      raise

  def _load_model_arguments(self, data_dirname: str, uuid: str) -> Dict[Tag, Any]:
    """Loads model arguments from .pkl file."""
    model_arguments_filename = shared_file_utils.get_model_arguments_filename(
          data_dirname, uuid)
    print(f'Loading model arguments from filename={model_arguments_filename}')
    try:
      with open(model_arguments_filename, 'rb') as f:
        model_arguments = pickle.load(f)
    except (FileNotFoundError, OSError) as e:
      print(f'Failed to load from filename={model_arguments_filename} with error={e}.')
      raise

    return model_arguments
