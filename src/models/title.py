from dataclasses import dataclass
from typing import Dict
from pydantic import BaseModel, Field
from .evaluation import TitleEvaluation

@dataclass
class GeneratedTitle:
    main_title: str
    sub_title: str
    evaluation: TitleEvaluation
    original_main_title: str
    original_sub_title: str

class RefinedTitles(BaseModel):
    main_title: str = Field(description="修正後のメインタイトル")
    sub_title: str = Field(description="修正後のサブタイトル")

    def model_dump(self) -> Dict[str, str]:
        return {
            "main_title": self.main_title,
            "sub_title": self.sub_title,
        }