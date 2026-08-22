from fastapi import APIRouter
from app.database import user_collection

router = APIRouter()

@router.get("/{username}")
async def get_profile(username: str):
    user = await user_collection.find_one(
        {"username": username},
        {"password": 0}
    )
    return user

@router.get("/count")
async def count_users():
    count = await user_collection.count_documents({})
    return {"total_users": count}

@router.get("/all")
async def list_users():
    users = await user_collection.find({}, {"password": 0}).to_list(length=100)
    return {"users": users}