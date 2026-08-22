from fastapi import APIRouter

from app.services.safepay import (
    create_payment_session,
    create_passport_token
)

router = APIRouter()


@router.post("/create-payment")
def payment():

    tracker = create_payment_session()

    passport = create_passport_token()

    return {
        "tracker": tracker,
        "passport": passport
    }