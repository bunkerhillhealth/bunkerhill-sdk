from typing import Any, TypedDict


class Inference(TypedDict):
  model_id: str
  study_instance_uid: str
  series_instance_uid: str
  segmentation_presigned_url: str
