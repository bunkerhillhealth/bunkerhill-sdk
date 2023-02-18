"""Fixtures for test_model.py"""

import grpc_testing
import pytest

from grpc import Server

from bunkerhill.examples.hippocampus.model import MSDHippocampusModel
from bunkerhill.model_runner import ModelRunner
from bunkerhill.proto.inference_pb2 import DESCRIPTOR


@pytest.fixture(scope="function")
def model() -> ModelRunner:
  return ModelRunner(model_release=MSDHippocampusModel())


@pytest.fixture(scope="function")
def grpc_server(model) -> Server:
  service = DESCRIPTOR.services_by_name['InferenceProcessor']
  return grpc_testing.server_from_dictionary({service: model}, grpc_testing.strict_real_time())

