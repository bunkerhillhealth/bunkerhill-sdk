"""Command to convert 3D NifTi arrays to input files compatible with Bunkerhill's ModelRunner.

Example usage:
  python nifti_to_modelrunner_input.py \
    --nifti_filename=/path/to/hippocampus_009_0000.nii.gz \
    --data_dirname=/tmp \
    --study_uuid=77d1b303-f8b2-4aca-a84c-6c102d3625e1 \
    --series_uuid=e57ac58e-c0e8-44ab-be7e-4d17b32f6a8f
"""

import argparse
import pickle

import SimpleITK as sitk

from bunkerhill import shared_file_utils


def main(args: argparse.Namespace) -> None:
  input_array = sitk.GetArrayFromImage(sitk.ReadImage(args.nifti_filename))
  pixel_array = {'pixel_array': {args.series_uuid: input_array}}

  input_filename = shared_file_utils.get_model_arguments_filename(
    args.data_dirname, args.study_uuid
  )
  with open(input_filename, 'wb') as f:
    pickle.dump(pixel_array, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--nifti_filename', help='Path to NifTi file', required=True, type=str)
  parser.add_argument(
    '--data_dirname', help='Directory where input file will be written.', required=True, type=str
  )
  parser.add_argument(
    '--study_uuid',
    help=('UUID identifier of study. Inputs will be encoded as '
          '{study_uuid}_input.pkl.'),
    required=True,
    type=str
  )
  parser.add_argument(
    '--series_uuid',
    help='UUID identifier of series from which pixel array input originates.',
    required=True,
    type=str
  )
  main(parser.parse_args())
