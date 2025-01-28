from openai import OpenAI
import json

from ...models import HeadlineSet
from ...utils.prompts import HEADLINE_GENERATION_PROMPT

class HeadlineGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fixed_prompt_part = """
「『{title}』というタイトルのイベントを企画しており、その告知文を作成します。 告知文を作成する前に、以下の内容でその見出しを３つ作成してください。それぞれの見出しは簡潔な文章としてください。 」
"""
        self.user_editable_prompt = HEADLINE_GENERATION_PROMPT
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください：
{
    "background": "背景の見出し",
    "problem": "課題の見出し",
    "solution": "解決策の見出し"
}
"""

    def generate_headlines(self, title: str, prompt_template: str = None) -> HeadlineSet:
        prompt = self.fixed_prompt_part.format(title=title) + (prompt_template or self.user_editable_prompt) + self.fixed_output_instructions

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            result_text = response.choices[0].message.content.strip()
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                if start_index != -1 and end_index > start_index:
                    json_text = result_text[start_index:end_index]
                    result = json.loads(json_text)

            return HeadlineSet.from_dict(result)

        except Exception as e:
            raise Exception(f"見出し生成でエラーが発生しました: {str(e)}")