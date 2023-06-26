import { IFailedRequestError, IInvalidInferenceAPIClientArgsException } from './interfaces';

export class InvalidInferenceAPIClientArgsException extends Error {
  constructor(args: IInvalidInferenceAPIClientArgsException) {
    super(args.reason);
    this.name = 'InvalidInferenceAPIClientArgsException';
  }
}

export class FailedRequestError extends Error {
  constructor(args: IFailedRequestError) {
    var message = '{method} {url} returned a status code of {status_code} with error: {error}'
      .replace('{method}', args.method?.toString() ?? 'undefined')
      .replace('{url}', args.url)
      .replace('{status_code}', args.response.status.toString())
      .replace('{error}', args.error)
    super(message)
    this.name = 'FailedRequestError'
  }
}
