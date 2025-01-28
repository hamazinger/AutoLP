from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class WebContent:
    title: str
    description: str
    main_content: str
    error: Optional[str] = None

@dataclass
class HeadlineSet:
    background: str
    problem: str
    solution: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "background": self.background,
            "problem": self.problem,
            "solution": self.solution
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'HeadlineSet':
        return cls(
            background=data.get("background", ""),
            problem=data.get("problem", ""),
            solution=data.get("solution", "")
        )