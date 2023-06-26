import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import * as jwt from 'jsonwebtoken';
import { FailedRequestError } from './exceptions'
import retry from 'async-retry';


interface ClientParams {
  username: string;
  clientPrivateKey: string;
  baseUrl: string;
  authPath: string;
  numFailuresAllowed?: number;
}

export default class DjangoJWTClient {
  private static readonly CLIENT_JWT_ENCODING_ALGORITHM = 'RS256';
  private static readonly AUTH_REQUEST_HEADERS = {
    'Content-Type': 'application/json',
  };

  private djangoBaseUrl: string;
  private authPath: string;
  private numFailures: number = 0;
  private numFailuresAllowed: number;
  private username: string;
  private clientPrivateKey: string;
  private axiosInstance: AxiosInstance;
  private accessJwt: string | null = null;

  constructor(params: ClientParams) {
    this.username = params.username;
    this.clientPrivateKey = params.clientPrivateKey;
    this.djangoBaseUrl = params.baseUrl;
    this.authPath = params.authPath;
    this.numFailuresAllowed = params.numFailuresAllowed || 3;
    this.axiosInstance = axios.create();
  }

  async getJson(path: string): Promise<any> {
    await this.ensureJwtAccess();
    return await this.makeRequest(path, 'GET', null, { Authorization: `Bearer ${this.accessJwt}` });    
  }

  private generateClientJwt(): string {
    return jwt.sign(
      { username: this.username },
      this.clientPrivateKey,
      { algorithm: DjangoJWTClient.CLIENT_JWT_ENCODING_ALGORITHM }
    );
  }

  private async handleFailedRequest(e: any, url: string, method: AxiosRequestConfig["method"]) {
    this.numFailures += 1;
    if (this.numFailures > this.numFailuresAllowed && url != this.djangoBaseUrl + this.authPath) {
      this.accessJwt = null;
      await this.ensureJwtAccess();
    }
    throw new FailedRequestError({
      url: url,
      method: method,
      response: e.response,
      error: e.toString()
    });
  }

  private async ensureJwtAccess() {
    if (!this.accessJwt) {
      const clientJwt = this.generateClientJwt();
      const authUrl = `${this.djangoBaseUrl}${this.authPath}`;
      const response = await this.makeRequest(
        authUrl,
        'POST',
        { token: clientJwt },
        { headers: DjangoJWTClient.AUTH_REQUEST_HEADERS }
      );
      if (response != undefined) {
        this.accessJwt = response.data.access;
      }
    }
  }

  private async makeRequest(path: string, method: AxiosRequestConfig["method"], data?: any, headers?: any): Promise<any> {
    const url = this.djangoBaseUrl + path;
    try {
      return await retry(async (bail: any, numberOfRetries: number) => {
        const result = await this.axiosInstance({
          method: method,
          url: url,
          headers: headers,
          data: data
        });
        if (result.status >= 400) {
          throw new Error(`Request failed with status code ${result.status}`);
        }
        return result;
      }, {
        retries: 3,
        minTimeout: 100,
      });
    } catch (e) {
      this.handleFailedRequest(e, url, method)
    }
  }
}