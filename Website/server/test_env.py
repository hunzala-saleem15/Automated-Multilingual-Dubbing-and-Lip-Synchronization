from dotenv import load_dotenv
import os

load_dotenv()

print("Public Key:", os.getenv("SAFEPAY_PUBLIC_KEY"))
print("Secret Key:", os.getenv("SAFEPAY_SECRET_KEY"))