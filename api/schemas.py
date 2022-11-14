from pydantic import BaseModel, EmailStr

#Auth
class UserLogIn(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    password: str

    class Config:
        orm_mode = True

#Feedback
class FeedbackCreate(BaseModel):
    feedback: str
    rating: int
    prediction: str

class FeedbackOut(BaseModel):
    id: int
    feedback: str
    rating: int
    prediction: str


    class Config:
        orm_mode = True