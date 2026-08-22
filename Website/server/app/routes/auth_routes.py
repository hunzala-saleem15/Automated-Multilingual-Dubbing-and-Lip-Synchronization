from fastapi import APIRouter, HTTPException, Body, Depends, UploadFile, File
from fastapi.responses import RedirectResponse
from datetime import datetime, date
import re
import httpx
import os
import shutil
from bson import ObjectId
from pathlib import Path

# Database connection
from app.database import user_collection, db
from app.schemas.user_schema import UserRegister, UserLogin
from app.utils.hash import hash_password, verify_password
from app.utils.jwt import create_access_token
from app.dependencies.auth import get_current_user

router = APIRouter()

# Directories Setup
UPLOAD_DIR = Path("uploads/profile_photos")
DUBBED_DIR = Path("uploads/dubbed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DUBBED_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. AUTHENTICATION (Register & Login)
# =========================

@router.post("/register")
async def register_user(user: UserRegister):
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_regex, user.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    if (len(user.password) < 8 or not re.search(r"[A-Za-z]", user.password) or not re.search(r"\d", user.password)):
        raise HTTPException(status_code=400, detail="Password must be 8+ chars with letters & numbers")

    existing_user = await user_collection.find_one({"$or": [{"email": user.email}, {"username": user.username}]})
    if existing_user:
        raise HTTPException(status_code=409, detail="User already exists")

    user_dict = {
        "username": user.username,
        "email": user.email,
        "password_hash": hash_password(user.password),
        "first_name": user.first_name,
        "last_name": user.last_name,
        "gender": user.gender,
        "dob": datetime.combine(user.dob, datetime.min.time()),
        "date_joined": datetime.utcnow(),
        "plan": "Free",
        "profile_photo": None,
        "saved_videos": [],
        "total_minutes_used": 0.0,
        "is_active": True,
    }

    result = await user_collection.insert_one(user_dict)
    token = create_access_token(data={"user_id": str(result.inserted_id), "email": user.email})
    return {"access_token": token, "token_type": "bearer"}

@router.post("/login")
async def login_user(user: UserLogin):
    db_user = await user_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token(data={"user_id": str(db_user["_id"]), "email": db_user["email"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": db_user["username"],
            "plan": db_user.get("plan", "Free"),
            "profile_photo": db_user.get("profile_photo")
        }
    }

# =========================
# 2. PROFILE & SETTINGS
# =========================

@router.get("/me")
async def get_current_user_data(current_user: dict = Depends(get_current_user)):
    user = await user_collection.find_one({"_id": ObjectId(current_user["user_id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["_id"] = str(user["_id"])
    if "password_hash" in user: del user["password_hash"]
    if "saved_videos" not in user: user["saved_videos"] = []
    return user

@router.post("/change-password")
async def change_password(new_password: str = Body(...), current_user: dict = Depends(get_current_user)):
    user = await user_collection.find_one({"_id": ObjectId(current_user["user_id"])})
    if verify_password(new_password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="New password cannot be same as old")
    
    await user_collection.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {"$set": {"password_hash": hash_password(new_password)}}
    )
    return {"message": "Password updated successfully"}

@router.post("/profile/photo")
async def upload_profile_photo(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    ext = Path(file.filename).suffix
    filename = f"{current_user['user_id']}{ext}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    db_path = f"uploads/profile_photos/{filename}"
    await user_collection.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {"$set": {"profile_photo": db_path}}
    )
    return {"profile_photo": db_path}

# =========================
# 3. VIDEO & PLAN MANAGEMENT
# =========================

@router.post("/save-video")
async def save_video(
    file: UploadFile = File(...), 
    video_name: str = Body(...),
    duration: float = Body(0.0),
    current_user: dict = Depends(get_current_user)
):
    ext = Path(file.filename).suffix
    unique_name = f"{current_user['user_id']}_{int(datetime.utcnow().timestamp())}{ext}"
    file_path = DUBBED_DIR / unique_name

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_entry = {
        "video_id": str(ObjectId()),
        "video_url": f"uploads/dubbed/{unique_name}",
        "video_name": video_name,
        "duration_mins": duration,
        "saved_at": datetime.utcnow()
    }

    await user_collection.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {
            "$push": {"saved_videos": video_entry},
            "$inc": {"total_minutes_used": duration}
        }
    )
    return {"message": "Video saved", "path": video_entry["video_url"]}

@router.post("/update-plan")
async def update_user_plan(payload: dict = Body(...), current_user: dict = Depends(get_current_user)):
    plan_type = payload.get("plan_type")
    await user_collection.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {"$set": {"plan": plan_type}}
    )
    return {"status": "success", "message": f"Plan updated to {plan_type}"}

# =========================
# 4. GOOGLE OAUTH
# =========================
GOOGLE_CLIENT_ID = os.getenv("179511982939-ce8o3a1k6da2ktajc0375fgo87csdame.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.getenv("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")
GOOGLE_REDIRECT_URI = "http://127.0.0.1:8000/auth/google/callback"

@router.get("/google/login")
async def google_login():
    url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={GOOGLE_REDIRECT_URI}&response_type=code&scope=email%20profile"
    return RedirectResponse(url)

@router.get("/google/callback")
async def google_callback(code: str):
    async with httpx.AsyncClient() as client:
        t_res = await client.post("https://oauth2.googleapis.com/token", data={
            "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code, "grant_type": "authorization_code", "redirect_uri": GOOGLE_REDIRECT_URI
        })
        user_info = (await client.get("https://www.googleapis.com/oauth2/v2/userinfo", 
                     headers={"Authorization": f"Bearer {t_res.json()['access_token']}"})).json()

    email = user_info["email"]
    db_user = await user_collection.find_one({"email": email})
    
    if not db_user:
        res = await user_collection.insert_one({
            "username": email.split("@")[0], "email": email, "plan": "Free",
            "date_joined": datetime.utcnow(), "saved_videos": [], "is_active": True
        })
        u_id = str(res.inserted_id)
    else:
        u_id = str(db_user["_id"])

    token = create_access_token(data={"user_id": u_id, "email": email})
    return {"access_token": token, "token_type": "bearer"}