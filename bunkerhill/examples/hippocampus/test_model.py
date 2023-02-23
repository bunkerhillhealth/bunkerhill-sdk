"""Test for model.py"""

import os
import pickle
import uuid

from pathlib import Path

import numpy as np

from grpc import Server, StatusCode

from bunkerhill import shared_file_utils
from bunkerhill.examples.hippocampus.model import MSDHippocampusModel
from bunkerhill.proto.inference_pb2 import DESCRIPTOR
from bunkerhill.proto.inference_pb2 import InferenceRequest
from bunkerhill.proto.inference_pb2 import InferenceResponse


def test_run_inference(tmp_path: Path, grpc_server: Server):
  # Create fake test instance and dump it to directory from where the model can read it.
  data_dirname = str(tmp_path)
  study_identifier = str(uuid.uuid4())
  pixel_array = np.random.randint(2, 165, (39, 47, 36), dtype=np.uint8)
  series_uid = '1.2.314159.117779'
  model_arguments = {'pixel_array': {series_uid: pixel_array}}
  model_arguments_filename = shared_file_utils.get_model_arguments_filename(
    data_dirname, study_identifier)
  with open(model_arguments_filename, 'wb') as f:
    pickle.dump(model_arguments, f)

  # Send an request to the model's gRPC server to run hippocampus segmentation.
  method_descriptor = DESCRIPTOR.services_by_name['InferenceProcessor'].methods_by_name['RunInference']
  request = InferenceRequest(data_dirname=data_dirname,
                             uuid=study_identifier)
  runner = grpc_server.invoke_unary_unary(method_descriptor=method_descriptor,
                                          invocation_metadata={},
                                          request=request,
                                          timeout=60)
  response, _, code, _ = runner.termination()
  assert response == InferenceResponse()
  assert code == StatusCode.OK

  # Read the segmentation and softmax arrays output by nnUNet.
  model_outputs_filename = shared_file_utils.get_model_outputs_filename(data_dirname,
                                                                        study_identifier)
  with open(model_outputs_filename, 'rb') as f:
    outputs = pickle.load(f)

  segmentation = outputs[MSDHippocampusModel._SEGMENTATION_OUTPUT_ATTRIBUTE_NAME][series_uid]
  assert segmentation.shape == (39, 47, 36)
  assert segmentation.dtype == np.uint8

  softmax = outputs[MSDHippocampusModel._SOFTMAX_OUTPUT_ATTRIBUTE_NAME][series_uid]
  assert softmax.shape == (3, 36, 47, 39)
  assert softmax.dtype == np.float16
