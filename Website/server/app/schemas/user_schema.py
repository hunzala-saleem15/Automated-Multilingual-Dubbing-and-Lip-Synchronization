from pydantic import BaseModel, EmailStr
from datetime import date

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    gender: str
    dob: date

class UserLogin(BaseModel):
    email: EmailStr
    password: str