// sample.ts
import 'source-map-support/register';
import { InferenceAPIClient } from './client';

async function main() {
  const username = 'datashare-admin';
  const privateKeyFilename = '/Users/gabealvarez/bunkerhill/bunkerhill/private_key.pem';
  const baseUrl = 'https://api.coppshillhealth.com/';

  const client = new InferenceAPIClient(username, privateKeyFilename, undefined, baseUrl);

  const modelId = 'e7ad8122-14cf-4bb2-b57f-075a07b51e2b';
  const patientMrn = '1';
  
  try {
    const inferences = await client.getInferencesAsync(
      modelId,
      patientMrn,
    );
    
    console.log(JSON.stringify(inferences, null, 2));
  } catch (error) {
    console.error(`Error while getting inferences: ${error}`);
  }
}

main();
