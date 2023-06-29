// Class definition for Inference API Client
import { AxiosInstance } from 'axios';
import fs from 'fs';
import DjangoJWTClient from './djangoJwtClient';
import { InvalidInferenceAPIClientArgsException } from './exceptions'
import { IInference } from './interfaces';

export class InferenceAPIClient {
  private static readonly AUTH_PATH: string = 'api/auth/jwt_login/';
  private static readonly GET_INFERENCE_PATH: string = 'api/models/{modelId}/patients/{patientMrn}/inferences/';

  private djangoJwtClient: DjangoJWTClient;

  constructor(
    username: string,
    privateKeyFilename?: string,
    privateKeyString?: string,
    baseUrl: string = 'https://api.bunkerhillhealth.com/',
  ) {
    /*
    Constructs an InferenceAPIClient.

    Args:
      username: Username for authentication.
      privateKeyFilename: Path to the private key file for authentication.
      privateKeyString: String representation of the private key for authentication.
      baseUrl: Base URL for the API. Defaults to 'https://api.bunkerhillhealth.com/'.
    Raises:
      InvalidInferenceAPIClientArgsException: If invalid arguments are passed to the constructor.
    */
    this.validateArgs(privateKeyFilename, privateKeyString);

    let clientPrivateKey = "";
    if (privateKeyFilename) {
      clientPrivateKey = fs.readFileSync(privateKeyFilename, 'utf-8');
    } else if (privateKeyString) {
      clientPrivateKey = privateKeyString;
    }

    this.djangoJwtClient = new DjangoJWTClient({
      username: username,
      clientPrivateKey: clientPrivateKey,
      baseUrl: baseUrl,
      authPath: InferenceAPIClient.AUTH_PATH
    });
  }

  private validateArgs(
    privateKeyFilename?: string,
    privateKeyString?: string,
  ): void {
    if (!privateKeyFilename && !privateKeyString) {
      throw new InvalidInferenceAPIClientArgsException({
        reason: 'either privateKeyFilename or privateKeyString must be provided',
      });
    }
    if (privateKeyFilename && privateKeyString) {
      throw new InvalidInferenceAPIClientArgsException({
        reason: 'privateKeyFilename and privateKeyString cannot both be provided',
      });
    }
  }

  public async getInferences(
    modelID: string,
    patientMrn: string,
  ): Promise<IInference[]> {
    /*
    Asynchronous function to fetch inferences from the API matching the modelId and patientMrn, and
    download their segmentations locally.

    Args:
      modelId: Model ID for the inferences.
      patientMrn: Medical record number (MRN) of the patient.

    Returns:
      List of Inferences fetched from the API.

    Raises:
      InferenceAPIRequestFailedError: If a 400- or 500- response, or a response with invalid data, 
      is received from the server.
    */
    const resourcePath = InferenceAPIClient.GET_INFERENCE_PATH
      .replace('{modelID}', modelId)
      .replace('{patientMrn}', patientMrn);
    const response = await this.djangoJwtClient.getJson(resourcePath);
    return response.data as IInference[];
  }
}
