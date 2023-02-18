"""API to convert data format between nnUNet and Bunkerhill Health."""

import dataclasses
import os

from typing import Dict

import nibabel as nib
import numpy as np

from bunkerhill.bunkerhill_types import SeriesUID

# See inference dataset naming conventions at:
# https://github.com/MIC-DKFZ/nnUNet/blob/7f1e273fa1021dd2ff00df2ada781ee3133096ef/documentation/data_format_inference.md
_MODALITY_SUFFIX = '0000'
_TEST_INSTANCE_ID = 'instance_0'


@dataclasses.dataclass(frozen=True)
class NNUNetPaths:
  """The paths required by nnUNet to run inference.

  See path documentation at:
  https://github.com/MIC-DKFZ/nnUNet/blob/aa53b3b87130ad78f0a28e6169a83215d708d659/documentation/setting_up_paths.md#setting-up-paths
  """
  raw_data_dirname: str
  preprocessed_dirname: str
  results_dirname: str
  test_data_dirname: str
  output_dirname: str


def setup_paths(data_dirname: str, task: str) -> NNUNetPaths:
  """Creates folders and environment variables for nnUNet workspace.

  Follows the official documentation on dataset paths and environment variables:
  https://github.com/MIC-DKFZ/nnUNet/blob/aa53b3b87130ad78f0a28e6169a83215d708d659/documentation/setting_up_paths.md#setting-up-paths

  Args:
    data_dirname: The base directory where all the nnUNet subdirectories will be created.
    task: The name of the nnUNet task. Tasks are encoded as Task{id}_{name} where id is a three
      digit integer and name is a string. Custom task IDs start at 500 to prevent conflicts with
      downloaded pretrained models, and IDs cannot exceed 999. See documentation at:
      https://github.com/MIC-DKFZ/nnUNet/blob/7f1e273fa1021dd2ff00df2ada781ee3133096ef/documentation/dataset_conversion.md

  Returns:
    All the paths created for nnUNet.
  """
  raw_data_dirname = os.path.join(data_dirname, 'nnUNet_raw_data_base')
  preprocessed_dirname = os.path.join(data_dirname, 'nnUNet_preprocessed')
  results_dirname = os.path.join(data_dirname, 'nnUNet_trained_models')
  output_dirname = os.path.join(data_dirname, 'nnUNet_output')
  os.makedirs(raw_data_dirname, exist_ok=True)
  os.makedirs(preprocessed_dirname, exist_ok=True)
  os.makedirs(results_dirname, exist_ok=True)
  os.makedirs(output_dirname, exist_ok=True)

  # Make imagesTs directory under nnUNet_raw_data_base
  test_data_dirname = os.path.join(raw_data_dirname, 'nnUNet_raw_data', task, 'imagesTs')
  os.makedirs(test_data_dirname, exist_ok=True)

  os.environ['nnUNet_raw_data_base'] = raw_data_dirname
  os.environ['nnUNet_preprocessed'] = preprocessed_dirname
  os.environ['RESULTS_FOLDER'] = results_dirname

  return NNUNetPaths(
    raw_data_dirname=raw_data_dirname,
    preprocessed_dirname=preprocessed_dirname,
    results_dirname=results_dirname,
    test_data_dirname=test_data_dirname,
    output_dirname=output_dirname
  )


def dump_pixel_array(pixel_array: np.ndarray, nnunet_input_dirname: str) -> None:
  """Converts pixel_array from NumPy ndarray to 3D NifTi file.

  nnUNet expects input images to be in 3D NifTi files, while Bunkerhill unpacks DICOM pixel_arrays
  as Numpy ndarrays. See nnUNet dataset documentation at:
  https://github.com/MIC-DKFZ/nnUNet/blob/7f1e273fa1021dd2ff00df2ada781ee3133096ef/documentation/dataset_conversion.md

  Args:
    pixel_array: The pixel_array spanning all instances in the same series.
    nnunet_input_dirname: The directory path where the 3D NifTi pixel_array will be written.
  """
  model_argument_filename = os.path.join(
    nnunet_input_dirname, f'{_TEST_INSTANCE_ID}_{_MODALITY_SUFFIX}.nii.gz'
  )
  nifti_pixel_array = nib.Nifti1Image(pixel_array, affine=np.eye(4))
  nib.save(nifti_pixel_array, model_argument_filename)


def load_segmentation(outputs_dirname: str, output_attribute_name: str,
                      series_uid: SeriesUID) -> Dict[str, Dict[str, np.ndarray]]:
  """Loads the nnUNet output segmentation tensor from a 3D NifTI file to an output attribute dict.

  Args:
    outputs_dirname: The directory path where nnUNet wrote inference output.
    output_attribute_name: The name of the segmentation output attribute. This is used by the
      Bunkerhill pipeline to identify a model's segmentation output.
    series_uid: The UID of the DICOM series on which nnUNet ran inference. See:
      https://dicom.innolitics.com/ciods/cr-image/general-series/0020000e

  Returns:
    A dictionary of output attributes. The dictionary has the format:
      {
        output_attribute_name:
          {
            series_uid: segmentation_ndarray
          }
      }
  """
  segmentation_ndarray = nib.load(os.path.join(outputs_dirname,
                                               f'{_TEST_INSTANCE_ID}.nii.gz')).get_data()
  return {output_attribute_name: {series_uid: segmentation_ndarray}}


def load_softmax(outputs_dirname: str, output_attribute_name: str,
                 series_uid: SeriesUID) -> Dict[str, Dict[str, np.ndarray]]:
  """Loads the nnUNet output softmax tensor from an npz file to an output attribute dict.

  Args:
    outputs_dirname: The directory path where nnUNet wrote inference output.
    output_attribute_name: The name of the segmentation output attribute. This is used by the
      Bunkerhill pipeline to identify a model's segmentation output.
    series_uid: The UID of the DICOM series on which nnUNet ran inference. See:
      https://dicom.innolitics.com/ciods/cr-image/general-series/0020000e

  Returns:
    A dictionary of output attributes. The dictionary has the format:
      {
        output_attribute_name:
          {
            series_uid: softmax_ndarray
          }
      }
  """
  softmax_ndarray = np.load(os.path.join(outputs_dirname, f'{_TEST_INSTANCE_ID}.npz'))['softmax']
  return {output_attribute_name: {series_uid: softmax_ndarray}}
