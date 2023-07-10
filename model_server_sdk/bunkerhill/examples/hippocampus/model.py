"""The class definition and model server entrypoint for the MSD Hippocampus model."""

import subprocess

from typing import Dict, Tuple

import numpy as np

from bunkerhill import image_utils
from bunkerhill import nnunet_wrapper
from bunkerhill.base_model import BaseModel
from bunkerhill.bunkerhill_types import Outputs, SeriesInstanceUID
from bunkerhill.nnunet_wrapper import NNUNetPaths
from bunkerhill.model_runner import ModelRunner


class MSDHippocampusModel(BaseModel):
  """A wrapper around the trained nnUNet model for the MSD Hippocampus model.

  This wrapper utilizes the nnUNet command line tools to load the model weights and run inference
  on each individual test instance (a DICOM series).

  For details about the Medical Segmentation Decathlon and the Hippocampus dataset, see:
  http://medicaldecathlon.com/

  The model weights are downloaded from:
  https://zenodo.org/record/4003545/files/Task004_Hippocampus.zip

  Attributes:
    _paths: The paths required by nnUNet to run inference.
  """

  _DATA_DIRNAME: str = '/data'
  _TASK_NAME: str = 'Task004_Hippocampus'
  _PRETRAINED_MODEL_FILENAME: str = f'/app/{_TASK_NAME}.zip'
  _SEGMENTATION_OUTPUT_ATTRIBUTE_NAME: str = 'hippocampus_seg_pred'
  _SOFTMAX_OUTPUT_ATTRIBUTE_NAME: str = 'hippocampus_softmax_pred'
  _LOAD_WEIGHTS_COMMAND: str = 'nnUNet_install_pretrained_model_from_zip'
  _INFERENCE_COMMAND: str = 'nnUNet_predict'
  _SAVE_SOFTMAX_FLAG: str = '--save_npz'

  _paths: NNUNetPaths

  def __init__(self):
    self._paths = nnunet_wrapper.setup_paths(self._DATA_DIRNAME, self._TASK_NAME)

    # Install the pretrained Hippocampus model from the weights bundled in the .zip file. The
    # weights are downloaded to the Docker image in the Dockerfile.
    install_pretrained_model_cmd = [self._LOAD_WEIGHTS_COMMAND, self._PRETRAINED_MODEL_FILENAME]
    subprocess.check_call(install_pretrained_model_cmd, timeout=300)

  def inference(
    self,
    image_position_patient: Dict[SeriesInstanceUID, Dict[int, Tuple[float, float, float]]],
    pixel_array: Dict[SeriesInstanceUID, np.ndarray],
    pixel_spacing: Dict[SeriesInstanceUID, Tuple[float, float]],
  ) -> Outputs:
    """Runs inference on the pixel array for a DICOM series.

    Args:
      image_position_patient: The x, y, and z coordinates of the upper left hand corner of each
        instance.
      pixel_array: A dict mapping the DICOM series UID to its pixel array.
      pixel_spacing: The pair of values specifying physical distance in the patient between the
        center of each pixel.

    Returns:
      A dictionary containing the output segmentation and softmax ndarrays.
    """
    pixel_spacing_z = image_utils.compute_z_dim_pixel_spacing(
      next(iter(image_position_patient.values())))
    pixel_spacing_x, pixel_spacing_y = next(iter(pixel_spacing.values()))
    first_series_pixel_array = next(iter(pixel_array.values()))

    # Convert Bunkerhill pipeline's model arguments into format expected by nnUNet.
    nnunet_wrapper.dump_pixel_array(
      [first_series_pixel_array],
      [(pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)],
      self._paths.test_data_dirname
    )

    # Run model inference using nnUNet_predict command line tool. Save the softmax tensor in
    # addition to the segmentation.
    inference_cmd_args = [
      self._INFERENCE_COMMAND, '-i', self._paths.test_data_dirname, '-o',
      self._paths.output_dirname, '-t', self._TASK_NAME, '-m', '3d_fullres', self._SAVE_SOFTMAX_FLAG
    ]
    subprocess.check_call(inference_cmd_args, timeout=300)

    # Convert nnUNet segmentation and softmax tensors into output attributes.
    series_uid = next(iter(pixel_array.keys()))
    segmentation_output_attribute = nnunet_wrapper.load_segmentation(
      self._paths.output_dirname, self._SEGMENTATION_OUTPUT_ATTRIBUTE_NAME, series_uid
    )
    softmax_output_attribute = nnunet_wrapper.load_softmax(
      self._paths.output_dirname, self._SOFTMAX_OUTPUT_ATTRIBUTE_NAME, series_uid
    )
    return {**segmentation_output_attribute, **softmax_output_attribute}


if __name__ == '__main__':
  model = MSDHippocampusModel()
  model_runner = ModelRunner(model)
  model_runner.start_run_loop()
