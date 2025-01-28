from openai import OpenAI

from ...models import HeadlineSet
from ...utils.prompts import BODY_GENERATION_PROMPT

class BodyGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fixed_prompt_part = """
以下のセミナータイトルと見出しに基づいて、本文を生成してください：
- 各見出しは本文中に明示してください。明確に見出しであることがわかるマークダウンの書式（見出しレベル4）を用いてください。

タイトル：「{title}」
{background}
{problem}
{solution}
"""
        self.user_editable_prompt = BODY_GENERATION_PROMPT

    def generate_body(self, title: str, headlines: HeadlineSet, prompt_template: str = None) -> str:
        prompt = self.fixed_prompt_part.format(
            title=title,
            background=headlines.background,
            problem=headlines.problem,
            solution=headlines.solution
        ) + (prompt_template or self.user_editable_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"本文生成でエラーが発生しました: {str(e)}")
