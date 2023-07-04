"""Utility methods for image processing."""

import logging


from typing import Dict, Tuple


logger = logging.getLogger(__name__)

def compute_z_dim_pixel_spacing(
    image_position_patient: Dict[int, Tuple[float, float, float]]
) -> float:
  """Computes the z dimension pixel spacing value from the Image Position (Patient) DICOM tag."""
  instance_indices = image_position_patient.keys()
  first_instance_index = min(instance_indices)
  last_instance_index = max(instance_indices)
  first_z_position = image_position_patient[first_instance_index][2]
  last_z_position = image_position_patient[last_instance_index][2]

  # Warn if there are missing instances.
  num_expected_instances = last_instance_index - first_instance_index + 1
  num_actual_instances = len(instance_indices)
  if num_actual_instances != num_expected_instances:
    logger.warning(
      'Expected %s instances, but received %s instead.',
      num_expected_instances,
      num_actual_instances
    )
  return abs(last_z_position - first_z_position) / (last_instance_index - first_instance_index)
