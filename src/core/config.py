import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "HydrAI-SWE"
    VERSION: str = "1.0.0"
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/hydrai_swe")
    
    # API settings
    API_PREFIX: str = "/api/v1"

settings = Settings()
