from .. import models, schemas, utils
from fastapi import Response, status, HTTPException, Depends, APIRouter, Request
from ..database import get_db
from sqlalchemy.orm import Session
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

templates = Jinja2Templates(directory="api/templates")

router = APIRouter(
    prefix = "/users",
    tags = ["Users"]
)

#---WE ARE GOING TO CODE DIFFERENT FUNCTIONS WHICH PERFORM THE USER CREATION---#




@router.post("/", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def login(user:schemas.UserLogIn, request: Request, db: Session = Depends(get_db)):

    if utils.user_restrictions(user.email, user.password) == True:
        pass

    #hash the password - user.password
    hashed_password = utils.hash_password(user.password)
    user.password = hashed_password
    
    #Adding info to db
    new_user = models.User(**user.dict()) 
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user

