from pydantic import BaseSettings, Field, AnyUrl, validator
from typing import Optional, List
import os

class Settings(BaseSettings):
    # General
    app_name: str = Field("AstroDash API", env="APP_NAME")
    environment: str = Field("production", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")

    # API
    api_prefix: str = Field("/api/v1", env="API_PREFIX")
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")

    # Database
    db_url: Optional[AnyUrl] = Field(None, env="DATABASE_URL")
    db_echo: bool = Field(False, env="DB_ECHO")

    # Storage
    storage_dir: str = Field("storage", env="STORAGE_DIR")
    user_model_dir: str = Field("app/astrodash_models/user_uploaded", env="USER_MODEL_DIR")

    # ML Model Paths
    dash_model_path: str = Field("app/astrodash_models/zeroZ/pytorch_model.pth", env="DASH_MODEL_PATH")
    transformer_model_path: str = Field("app/astrodash_models/yuqing_models/TF_wiserep_v6.pt", env="TRANSFORMER_MODEL_PATH")

    # Logging
    log_dir: str = Field("logs", env="LOG_DIR")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Security
    secret_key: str = Field("supersecret", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(60 * 24, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Other
    osc_api_url: str = Field("https://api.sne.space", env="OSC_API_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("allowed_hosts", "cors_origins", pre=True)
    def split_str(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

def get_settings() -> Settings:
    return Settings()
