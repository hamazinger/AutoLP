from pydantic import BaseModel, BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    GCP_SERVICE_ACCOUNT: Dict[str, Any]
    PROJECT_NAME: str = "セミナータイトルジェネレーター"
    MODEL_NAME: str = "gpt-4o"
    MAX_TITLE_LENGTH: int = 40
    CATEGORIES_QUERY: str = """
    SELECT
        Seminar_Title,
        Acquisition_Speed,
        Major_Category,
        Category,
        Total_Participants,
        Action_Response_Count,
        Action_Response_Rate,
        User_Company_Percentage,
        Non_User_Company_Percentage
    FROM `mythical-envoy-386309.majisemi.majisemi_seminar_usukiapi`
    WHERE Seminar_Title IS NOT NULL
    AND Acquisition_Speed IS NOT NULL
    """

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

    @property
    def page_config(self) -> dict:
        return {
            "page_title": self.PROJECT_NAME,
            "layout": "wide"
        }

    @property
    def hide_streamlit_style(self) -> str:
        return """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """

    @property
    def openai_config(self) -> dict:
        return {
            "api_key": self.OPENAI_API_KEY,
            "model": self.MODEL_NAME
        }

    @property
    def bigquery_credentials(self) -> Dict[str, Any]:
        return self.GCP_SERVICE_ACCOUNT