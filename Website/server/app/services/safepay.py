import os
import requests

BASE_URL = "https://sandbox.api.getsafepay.com"

PUBLIC_KEY = os.getenv("SAFEPAY_PUBLIC_KEY")
SECRET_KEY = os.getenv("SAFEPAY_SECRET_KEY")

def create_payment_session():

    url = f"{BASE_URL}/order/payments/v3/"

    headers = {
        "Authorization": f"Bearer {SECRET_KEY}",
        "Content-Type": "application/json"
    }

    payload = {

        "merchant_api_key": PUBLIC_KEY,

        "intent": "CYBERSOURCE",

        "mode": "payment",

        "entry_mode": "raw",

        "currency": "PKR",

        "amount": 10000,

        "metadata": {
            "order_id": "ORDER_1001"
        }

    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)

    return response.json()

def create_passport_token():

    url = f"{BASE_URL}/client/passport/v1/token"

    headers = {
        "Authorization": f"Bearer {SECRET_KEY}"
    }

    response = requests.post(url, headers=headers)

    print(response.text)

    return response.json()

