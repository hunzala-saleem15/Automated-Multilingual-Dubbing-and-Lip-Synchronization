from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import (
    auth_routes,
    user_routes,
    video_routes,
    payment_routes,
)

app = FastAPI(
    title="Multilingual Dubbing API"
)

# -----------------------------------
# Static Files
# -----------------------------------

app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

# -----------------------------------
# CORS
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Routers
# -----------------------------------

app.include_router(
    auth_routes.router,
    prefix="/auth",
    tags=["Auth"]
)

app.include_router(
    user_routes.router,
    prefix="/users",
    tags=["Users"]
)

# ✅ Remove extra "/videos" prefix
app.include_router(
    video_routes.router
)

app.include_router(
    payment_routes.router,
    prefix="/payment",
    tags=["Payment"]
)

# -----------------------------------
# Health Check
# -----------------------------------

@app.get("/")
def root():
    return {
        "status": "Backend running successfully 🚀"
    }