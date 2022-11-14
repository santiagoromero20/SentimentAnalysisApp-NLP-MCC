from pydantic import BaseSettings

class Settings(BaseSettings):
    
    #PostreSQL
    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    database_username: str

    #REDIS
    REDIS_QUEUE: str
    REDIS_PORT: str 
    REDIS_DB_ID: int 
    REDIS_IP: str 
    API_SLEEP: float
    SERVER_SLEEP: float
    

    class Config:
        env_file = ".env"


settings = Settings()