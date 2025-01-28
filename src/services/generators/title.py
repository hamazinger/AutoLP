from typing import List, Dict, Optional
import json
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from ...models import RefinedTitles
from ...utils.prompts import TITLE_GENERATION_PROMPT
from ..extractor import URLContentExtractor

class TitleGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.url_extractor = URLContentExtractor()
        self.user_editable_prompt = TITLE_GENERATION_PROMPT
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください：
{
    "titles": [
        {
            "main_title": "メインタイトル1",
            "sub_title": "サブタイトル1"
        },
        {
            "main_title": "メインタイトル2",
            "sub_title": "サブタイトル2"
        },
        {
            "main_title": "メインタイトル3",
            "sub_title": "サブタイトル3"
        }
    ]
}
"""

    def generate_titles(self, context: str, prompt_template: str = None, product_url: str = None, file_content: str = None) -> List[Dict[str, str]]:
        additional_context = ""
        if product_url:
            content = self.url_extractor.extract_with_trafilatura(product_url)
            if content and not content.error:
                additional_context += f"""
製品タイトル: {content.title}
製品説明: {content.description}
製品詳細: {content.main_content[:1000]}
"""

        if file_content:
            additional_context += f"""
アップロードされたファイルの内容:
{file_content}
"""

        prompt = f"""
# 入力情報
{context}
{additional_context}
""" + (prompt_template or self.user_editable_prompt) + self.fixed_output_instructions

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
            result = json.loads(result_text)

            if not isinstance(result, dict) or "titles" not in result:
                raise ValueError("不正な応答形式です")

            titles = result["titles"]
            if not isinstance(titles, list) or not titles:
                raise ValueError("タイトルが見つかりません")

            return titles[:3]

        except Exception as e:
            raise Exception(f"タイトル生成でエラーが発生しました: {e}")

    def refine_title(self, main_title: str, sub_title: str, prompt: str) -> Optional[RefinedTitles]:
        parser = PydanticOutputParser(pydantic_object=RefinedTitles)

        prompt_template = PromptTemplate(
            template="以下のメインタイトルとサブタイトルを修正してください。\n{format_instructions}\nメインタイトル: {main_title}\nサブタイトル: {sub_title}\n要望: {prompt}",
            input_variables=["main_title", "sub_title", "prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        llm = ChatOpenAI(temperature=0, model=self.model, openai_api_key=self.client.api_key)
        chain = prompt_template | llm | parser

        try:
            output = chain.invoke({
                "main_title": main_title,
                "sub_title": sub_title,
                "prompt": prompt
            })
            return output
        except Exception as e:
            raise Exception(f"タイトル修正でエラーが発生しました: {e}")