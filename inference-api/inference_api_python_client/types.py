"""Data types used in the Inference API Client package"""

from typing import List, TypedDict


class Inference(TypedDict):
  """Stores data relating to a single inference returned by the Inference API."""
  model_id: str
  patient_mrn: str
  segmentation_presigned_urls: List[str]
