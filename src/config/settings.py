import os
from dataclasses import dataclass

@dataclass
class Settings:
    MODEL_NAME: str = "gpt-4o"
    MAX_TITLE_LENGTH: int = 40
    MAX_BODY_LENGTH: int = 1000
    MIN_SECTION_LENGTH: int = 300
    MAX_SECTIONS: int = 3
    
    # プロンプト設定
    SYSTEM_ROLE: str = "あなたは優秀なコピーライターです。"
    
    # 環境変数
    TRAFILATURA_USE_SIGNAL: str = "false"
    
    @classmethod
    def setup(cls):
        os.environ["TRAFILATURA_USE_SIGNAL"] = cls.TRAFILATURA_USE_SIGNAL