from passlib.context import CryptContext
from langdetect import detect
from fastapi import status, HTTPException
from model import text_normalizer

#Hash Password
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password):
    return pwd_context.hash(password)

def verify(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


#Feedback Cleaning
def clean(text):
    
    #Text Depuration
    text = text_normalizer.clean_text(
        text=text,
        puncts=True,
        stopwords=True,
        urls=True,
        emails=True,
        numbers=True,
        emojis=True,
        special_char=True,
        phone_num=True,
        non_ascii=True,
        multiple_whitespaces=True,
        contractions=True,
        currency_symbols=True,
        custom_pattern=None,
    )

    #Text lemattization
    text = text_normalizer.lemmatize_text(text)

    return text


#Feedback Restrictions
def check_language(feedback):
    flag = False
    language = detect(feedback)
    
    if language == 'en':
        flag = True
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail= "Please write your feedback in English!") 

    return flag

def check_length(feedback):
    flag = False
    words = len(feedback.split())
    if words <= 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Feedback must be greater than 5 words")
    elif words >= 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Feedback can not be greater than 100 words")
    else:
        flag = True

    return flag

def feedback_restrictions(feedback):
    return check_length(feedback) and check_language(feedback) 

def rating_restrictions(rating):
    flag = False
    valid_ratings = [1,2,3,4,5]

    for val_rating in valid_ratings:
        if val_rating == rating:
            flag = True
    
    if flag == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Rating value must be between 1 and 5")
    return flag

def restrictions(feedback, rating):
    return feedback_restrictions(feedback) and rating_restrictions(rating)

#User Restrictions
def email_restrictions(email):
    flag = False
    letters = list(email)
    for letter in letters:
        if letter != "@":
            flag = True
    
    if flag == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Invalid Email Address")
    return flag

def password_restrictions(password):
    flag = False
    letters = list(password)
    if len(letters) > 5:
            flag = True
    
    if flag == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Password must contain 6 or more characters")
    return flag

def user_restrictions(email, password):
    return email_restrictions(email) and password_restrictions(password)