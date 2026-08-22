from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# Collections
user_collection = db["users"]
video_collection = db["videos"]

# Optional: quick test
if __name__ == "__main__":
    print(f"✅ MongoDB connected: {DB_NAME}")