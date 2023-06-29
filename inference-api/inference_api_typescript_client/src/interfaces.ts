// Interface definitions used in Inference API Client module.
import { AxiosRequestConfig, AxiosResponse } from 'axios';

export interface IInvalidInferenceAPIClientArgsException {
  reason: string;
}

export interface IFailedRequestError {
  url: string;
  method: AxiosRequestConfig["method"];
  response: AxiosResponse<any>;
  error: string;
}

export interface IInference {
  [key: string]: any;
}
