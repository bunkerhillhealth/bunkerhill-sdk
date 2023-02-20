# Bunkerhill Health SDK

This repository contains an SDK for researchers to use as they prepared to deploy
their models on the [Bunkerhill Health](https://www.bunkerhillhealth.com/) inference platform.

## Table of contents

1. [Overview](#overview)
2. [Deploying your model](#deploying-your-model)
3. [ModelRelease](#modelrunner)
4. [Example model: hippocampus segmentation](#example-model-hippocampus-segmentation)
5. [Inputs](#inputs)
6. [Outputs](#outputs)

## Overview

Before your code can be transferred to Bunkerhill, you must first wrap your model using this SDK
to ensure Bunkerhill can run your model.

## Deploying your model

This SDK supports models running Python >=3.9.

You'll need the following components to run model inference on Bunkerhill:

- A [`requirements.txt`](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
file to download your model's [PyPI](https://pypi.org/) dependencies
- A Dockerfile to hermetically build your model with its dependencies in a Docker image. The
[hippocampus example Dockerfile](bunkerhill/examples/hippocampus/Dockerfile) provides an example
that include CUDA drivers on an Ubuntu 22.04 image.
- Test cases to assess model correctness. We also ask that you transfer test data so Bunkerhill
can continue to measure correctness throughout deployment. An example of model tests is provided
for the hippocampus example model at
[bunkerhill/examples/hippocampus/test_model.py](bunkerhill/examples/hippocampus/test_model.py).
- A model class to encapsulate all necessary steps for inference. This class must extend
[`BaseModel`](bunkerhill/base_model.py) and will also contain the entrypoint to runs your model via a
`ModelRunner`. An example of a minimal model class is shown below:

```python
from typing import Dict

import numpy as np

from tensorflow import keras

from bunkerhill.bunkerhill_types import OutputAttributes, SeriesUID
from bunkerhill.model import BaseModel
from bunkerhill.model_runner import ModelRunner

class MyModel(BaseModel):

  def __init__(self):
    # This path must be valid within the Docker container.
    self.model = keras.models.load_model('/path/to/weights')

  def inference(self, pixel_array: Dict[SeriesUID, np.ndarray]) -> OutputAttributes:
    """Runs inference on an entire DICOM series.

    Args:
      pixel_array: A Dict mapping the DICOM series UID to the 3D tensor of concatenated PixelData
        data elements.

    Returns:
      Dict containing the single output value
    """
    output = self.model(next(iter(pixel_array.values())))
    return {'output': output}


if __name__ == '__main__':
  model = MyModel()
  model_runner = ModelRunner(model)
  model_runner.start_run_loop()
```

## `ModelRunner`

[`ModelRunner`](bunkerhill/model_runner.py) hosts each model in a gRPC server and runs inference
in response to client requests. The sequence of interactions between client and server are as
follows:

1. Client saves pickled inference arguments to a file
2. Client sends an [`InferenceRequest`](bunkerhill/proto/inference.proto#L14) to `ModelRunner`
3. Upon receipt of the `InferenceRequest`, `ModelRunner` loads the arguments, runs inference, and
saves the pickled output to a file
4. `ModelRunner` sends an [`InferenceResponse`](bunkerhill/proto/inference.proto#L23) back to the
client

## Example model: hippocampus segmentation

The [hippocampus segmentation](bunkerhill/examples/hippocampus/model.py) model demonstrates how to
make an [nnUNet](https://github.com/MIC-DKFZ/nnUNet) model compatible with the Bunkerhill SDK. It
contains a segmentation model trained on the hippocampus dataset from the
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) using the nnUNet framework.
Inference is performed using the `nnUNet_predict` command line tool, and inputs and outputs are
converted between NumPy and NifTi formats.

### Build image

To build the example model, run
```shell
docker build \
  --build-arg USER_ID=$(id -u) \
  -t hippocampus:latest \
  . \
  -f bunkerhill/examples/hippocampus/Dockerfile
```

### Run unit tests

To run the hippocampus model unit tests, run:
```shell
export DATA_DIRNAME=/tmp
docker run -it \
  --mount type=bind,source=${DATA_DIRNAME},target=/data \
  hippocampus \
  pytest bunkerhill
```

### Interact with `ModelRunner` server

#### Start server

To run the hippocampus model as a server awaiting `InferenceRequest` messages, run:
```shell
export DATA_DIRNAME=/tmp
docker run -it \
  --mount type=bind,source=${DATA_DIRNAME},target=/data \
  hippocampus \
  python bunkerhill/examples/hippocampus/model.py
```

#### Generate inference input

For this example model, the
[`nifti_to_modelrunner_input.py`](bunkerhill/utils/nifti_to_modelrunner_input.py) utility can be
run to convert a NifTi image from the Medical Segmentation Decathlon into the necessary input
format.

First, download the hippocampus dataset from the
[Google Drive](https://drive.google.com/file/d/1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C/view?usp=share_link)
hosted by the Medical Segmentation Decathlon.

Once `Task04_Hippocampus.tar` has been unpacked, run the following command to convert one of NifTi
files into a `ModelRelease` input file:
```shell
export NIFTI_FILENAME=/path/to/Task004_Hippocampus/imagesTs/hippocampus_002_0000.nii.gz
export DATA_DIRNAME=/tmp
export STUDY_UUID=77d1b303-f8b2-4aca-a84c-6c102d3625e1
export SERIES_UUID=e57ac58e-c0e8-44ab-be7e-4d17b32f6a8f
python nifti_to_modelrunner_input.py \
  --nifti_filename=${NIFTI_FILENAME} \
  --data_dirname=${DATA_DIRNAME} \
  --study_uuid=${STUDY_UUID} \
  --series_uuid=${SERIES_UUID}
```

Note: the DICOM study and series UUIDs chosen above are arbitrary and can be replaced with any unique
identifiers.

For each study with UUID `${STUDY_UUID}`, the above script will generate a model argument file as a
pickled  dictionary of inputs named `${STUDY_UUID}_input.pkl`. After running inference, the
`ModelRunner` will return the outputs as a pickled dictionary named `${STUDY_UUID}_output.pkl`.

#### Send `InferenceRequest` messages

Once the server has started, `client_cli.py` can send `InferenceRequest` messages to the `ModelRunner`:
```shell
export STUDY_UUID=77d1b303-f8b2-4aca-a84c-6c102d3625e1
python client_cli.py \
  --socket_dirname=${DATA_DIRNAME} \
  --mounted_data_dirname=/data \
  --study_uuid=${STUDY_UUID}
```

Once inference has finished, the `ModelRunner` will write `${STUDY_UUID}_output.pkl` to the mounted
filesystem path and send an `InferenceResponse` back to the client.

## Inputs

Inputs to your model's `inference()` method can either be specific to the DICOM study or a specific
series in the study. Inputs specific to a series are passed to `inference()` as a dictionary
mapping the DICOM series UUID to the input value. Alternatively, inputs specific to a study are
just passed to `inference()` by their value.

### `PixelArray`

DICOM files store their pixel data in the
[`PixelData`](https://dicom.innolitics.com/ciods/segmentation/image-pixel/7fe00010) attribute.
Bunkerhill accesses this data via the
[pydicom `PixelArray` API](https://pydicom.github.io/pydicom/dev/old/working_with_pixel_data.html).

For a given DICOM series, Bunkerhill concatentates each slice's array into a
single 3D array.

### Non-`PixelArray` inputs

In addition to `PixelArray`, other DICOM attributes can be used as model inputs. These input
attributes can be either be defined in the DICOM spec or can be defined as
[private data elements](https://pydicom.github.io/pydicom/dev/old/private_data_elements.html).

Models deployed on Bunkerhill can currently include the following DICOM data elements as inputs to
inference.

- Age ([`PatientBirthDate`](https://dicom.innolitics.com/ciods/cr-image/patient/00100030) at
[`StudyDate`](https://dicom.innolitics.com/ciods/cr-image/general-study/00080020))
- [`Laterality`](https://dicom.innolitics.com/ciods/video-photographic-image/general-series/00200060)
- [`PatientBirthDate`](https://dicom.innolitics.com/ciods/rt-dose/patient/00100030)
- [`PatientSex`](https://dicom.innolitics.com/ciods/arterial-pulse-waveform/patient/00100040)
- [`PhotometricInterpretation`](https://dicom.innolitics.com/ciods/cr-image/cr-image/00280004)
- [`RescaleIntercept`](https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281052)
- [`RescaleSlope`](https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281053)
- [`SliceThickness`](https://dicom.innolitics.com/ciods/rt-dose/image-plane/00180050)
- [`WindowCenter`](https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281050)
- [`WindowWidth`](https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281051)

Other input attributes can be added to support new models. Please contact us if your model requires
other input attributes for inference.

## Output attributes

`inference()` can return any type(s) of output attributes. Examples include:

- Segmentation arrays
- Softmax arrays
- Predicted classifications
- Predicted scores

