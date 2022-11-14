from .. import models, schemas, utils
from fastapi import status, Depends, APIRouter
from ..database import get_db
from sqlalchemy.orm import Session
import joblib

router = APIRouter(
    prefix = "/feedback",
    tags = ["feedback"]
)

#Full Pipeline (Vectorizing and Predicting)
with open('spotify_pipe.pkl', 'rb') as f:
    loaded_pipe = joblib.load(f)

#Model Prediction
def predict_sentiment(text, loaded_pipe):
    return loaded_pipe.predict([text])[0]

#---WE ARE GOING TO CODE DIFFERENT FUNCTIONS WHICH PERFORM THE FEEDBACK TABLE CREATION---#

@router.post("/", status_code=status.HTTP_201_CREATED, response_model=schemas.FeedbackOut)
def create_feedback(feedback:schemas.FeedbackCreate,db: Session = Depends(get_db)):

    #Restriction Check
    if utils.restrictions(feedback.feedback, feedback.rating) == True:
        pass
    
    #Cleaning the Data
    feedback.feedback = utils.clean(feedback.feedback)
    
    #Getting Prediction
    feedback.prediction = predict_sentiment(feedback.feedback, loaded_pipe)
    
    #Saving Feedback into db for future retraining
    new_feedback = models.Feedback(**feedback.dict())
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)

    return new_feedback 




