// Custom exceptions raised in Inference API Client module.
import { IFailedRequestError, IInvalidInferenceAPIClientArgsException } from './interfaces';

export class InvalidInferenceAPIClientArgsException extends Error {
  // Exception raised when invalid args are passed to InferenceAPIClient constructor
  constructor(args: IInvalidInferenceAPIClientArgsException) {
    super(args.reason);
    this.name = 'InvalidInferenceAPIClientArgsException';
  }
}

export class FailedRequestError extends Error {
  // Exception raised when a request to the Inference API server fails.
  constructor(args: IFailedRequestError) {
    var message = '{method} {url} returned a status code of {status_code} with error: {error}'
      .replace('{method}', args.method?.toString() ?? 'undefined')
      .replace('{url}', args.url)
      .replace('{status_code}', args.response.status.toString())
      .replace('{error}', args.response.data)
    super(message)
    this.name = 'FailedRequestError'
  }
}
