from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from . import models
from .database import engine
from .routers import auth, feedback

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

models.Base.metadata.create_all(bind=engine)

#App Object Declaration
app = FastAPI()

origins=["*"]
#Add Middle
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, #Requests Allowed
    allow_credentials=True,
    allow_methods=["*"], #Methods Allowed
    allow_headers=["*"]  #Headers Allowed
)

#Importing Routers
app.include_router(auth.router)
app.include_router(feedback.router)




