"""The class definition and model server entrypoint for the MonaiFlexibleUNet model."""

import os

from typing import Dict

import numpy as np
import torch

from monai.networks.nets import FlexibleUNet

from bunkerhill.base_model import BaseModel
from bunkerhill.bunkerhill_types import Outputs, SeriesInstanceUID
from bunkerhill.model_runner import ModelRunner


class MonaiFlexibleUNet(BaseModel):
  """A wrapper around the trained nnUNet model for the MSD Hippocampus model.

  Attributes:
    _model: The pretrained PyTorch model to call at inference time.
  """
  _SEGMENTATION_OUTPUT_ATTRIBUTE_NAME: str = 'unet_seg_pred'

  def __init__(self):
    # Set model directory where pretrained model weights will be downloaded and unpacked.
    torch.hub.set_dir('/app')

    # Try loading a standard Torch model from a pth.tar file downloaded from HuggingFace Hub
    # Try compiling that model with torch.compile() for speed
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = FlexibleUNet(
      in_channels=1, out_channels=1, backbone='efficientnet-b0', pretrained=True
    )

    # Move model to GPU if available
    self.model.to(self.device)

    # Set model to eval mode
    self.model.eval()

  def inference(self, pixel_array: Dict[SeriesInstanceUID, np.ndarray]) -> Outputs:
    """Runs inference on the pixel array for a DICOM series.

    Args:
      pixel_array: A dict mapping the DICOM series UID to its pixel array.

    Returns:
      A dictionary containing the output segmentation and softmax ndarrays.
    """
    # Ensure pixel_array dict only containers a single series.
    if len(pixel_array) > 1:
      raise ValueError(f'Model only accepts a single series. {len(pixel_array)} were passed.')

    # Convert series pixel array from np.ndarray array to torch.Tensor.
    series_instance_uid = list(pixel_array.keys())[0]
    series_pixel_array = torch.from_numpy(pixel_array[series_instance_uid])

    # Move series_pixel_array to GPU if available and cast dtype to float32
    series_pixel_array = series_pixel_array.to(self.device, dtype=torch.float32)

    # Add batch dimension to series_pixel_array
    series_pixel_array = series_pixel_array.unsqueeze(0)

    # Run inference
    with torch.no_grad():
      segmentation = self.model(series_pixel_array)

    # Resize segmentation, move it to CPU, convert dtype to int16, and convert to ndarray.
    segmentation = segmentation.squeeze().to('cpu', dtype=torch.int16).numpy()

    # Convert nnUNet segmentation and softmax tensors into output attributes.
    return {self._SEGMENTATION_OUTPUT_ATTRIBUTE_NAME: {series_instance_uid: segmentation}}


if __name__ == '__main__':
  model = MonaiFlexibleUNet()
  model_runner = ModelRunner(model)
  model_runner.start_run_loop()
