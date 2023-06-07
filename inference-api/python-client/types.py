from typing import Any, TypedDict


class Inference(TypedDict):
  model_id: str
  patient_mrn: str
  segmentation_presigned_urls: List[str]
