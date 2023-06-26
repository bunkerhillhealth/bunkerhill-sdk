// InferenceAPIClient.ts
import { AxiosInstance } from 'axios';
import fs from 'fs';
import DjangoJWTClient from './django_jwt_client';
import { InvalidInferenceAPIClientArgsException } from './exceptions'
import { IInference } from './interfaces';

export class InferenceAPIClient {
  private static readonly AUTH_PATH: string = 'api/auth/jwt_login/';
  private static readonly GET_INFERENCE_PATH: string = 'api/models/{model_id}/patients/{patient_mrn}/inferences/';

  private djangoJwtClient: DjangoJWTClient;

  constructor(
    username: string,
    privateKeyFilename?: string,
    privateKeyString?: string,
    baseUrl: string = 'https://api.bunkerhillhealth.com/',
  ) {
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

  public async get_inferences_async(
    model_id: string,
    patient_mrn: string,
  ): Promise<IInference[]> {
    const resourcePath = InferenceAPIClient.GET_INFERENCE_PATH
      .replace('{model_id}', model_id)
      .replace('{patient_mrn}', patient_mrn);

    try {
      const response = await this.djangoJwtClient.getJson(resourcePath);
      return response.data as IInference[];
    } catch (error) {
      // Handle error appropriately. You can throw an exception or return a default value.
      console.error(error);
      return [];
    }
  }

  // Methods removed for brevity. Implement these based on your actual business logic.
}
