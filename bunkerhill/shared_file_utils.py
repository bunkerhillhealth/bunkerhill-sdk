"""Utilities for working with files read by and outputted by the ModelRunner."""


import os


def get_model_arguments_filename(dirname: str, uuid: str) -> str:
  """Returns the path to where on the Pod the pickled model arguments will be saved.

  Args:
    dirname: The directory name of the path to where the model arguments are saved.
    uuid: The UUID of the entity.

  Returns:
    The filename of the pickled model arguments on the Pod.
  """
  return os.path.join(dirname, f'{uuid}_input.pkl')

def get_model_outputs_filename(dirname: str, uuid: str) -> str:
  """Returns the path to where on the Pod the pickled model outputs will be saved.

  Args:
    dirname: The directory name of the path to where the model outputs are saved.
    uuid: The UUID of the entity.

  Returns:
    The filename of the pickled model outputs on the Pod.
  """
  return os.path.join(dirname, f'{uuid}_output.pkl')
