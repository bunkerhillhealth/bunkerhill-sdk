"""Test for model.py"""

import os
import pickle
import tempfile
import unittest
import uuid

import grpc_testing
import numpy as np

from grpc import Server, StatusCode

from bunkerhill import shared_file_utils
from bunkerhill.examples.monai.model import MonaiFlexibleUNet
from bunkerhill.model_runner import ModelRunner
from bunkerhill.proto.inference_pb2 import DESCRIPTOR
from bunkerhill.proto.inference_pb2 import InferenceRequest
from bunkerhill.proto.inference_pb2 import InferenceResponse


class TestMonaiFlexibleUNet(unittest.TestCase):

  def setUp(self):
    super().setUp()

    model = ModelRunner(MonaiFlexibleUNet())
    service = DESCRIPTOR.services_by_name['InferenceProcessor']
    self.grpc_server = grpc_testing.server_from_dictionary(
      {service: model}, grpc_testing.strict_real_time()
    )

  def test_run_inference(self):
    # Create fake test instance and dump it to directory from where the model can read it.
    study_identifier = str(uuid.uuid4())
    pixel_array = np.random.randint(low=0, high=256, size=(1, 512, 512))
    series_uid = '1.2.314159.117779'
    model_arguments = {'pixel_array': {series_uid: pixel_array}}
    with tempfile.TemporaryDirectory() as tmpdirname:
      model_arguments_filename = shared_file_utils.get_model_arguments_filename(
        tmpdirname, study_identifier
      )
      with open(model_arguments_filename, 'wb') as f:
        pickle.dump(model_arguments, f)

      # Send an request to the model's gRPC server to run MonaiFlexibleUNet segmentation.
      method_descriptor = DESCRIPTOR.services_by_name['InferenceProcessor'].methods_by_name[
        'RunInference']
      request = InferenceRequest(data_dirname=tmpdirname, uuid=study_identifier)
      runner = self.grpc_server.invoke_unary_unary(
        method_descriptor=method_descriptor, invocation_metadata={}, request=request, timeout=60
      )
      response, _, code, _ = runner.termination()
      assert response == InferenceResponse()
      assert code == StatusCode.OK

      # Read the segmentation output.
      model_outputs_filename = shared_file_utils.get_model_outputs_filename(
        tmpdirname, study_identifier
      )
      with open(model_outputs_filename, 'rb') as f:
        outputs = pickle.load(f)

    segmentation = outputs[MonaiFlexibleUNet._SEGMENTATION_OUTPUT_ATTRIBUTE_NAME][series_uid]
    self.assertEqual(segmentation.shape, (512, 512))
    self.assertEqual(segmentation.dtype, np.int16)


if __name__ == '__main__':
  unittest.main()
