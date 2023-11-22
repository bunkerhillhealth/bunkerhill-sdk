import asyncio

from bunkerhill_inference_api import InferenceAPIClient


async def main():
    async with InferenceAPIClient(
        username="datashare-admin",
        private_key_filename="private_key.pem",
        base_url="https://api.coppshillhealth.com/",
    ) as client:
        inference_list = await client.get_inferences(
            model_id="e7ad8122-14cf-4bb2-b57f-075a07b51e2b",
            patient_mrn="1",
        )
        print(inference_list)


asyncio.run(main())
