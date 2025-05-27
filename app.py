import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any  # Anyを追加
from datetime import datetime
import json
import requests
import re
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract
from PyPDF2 import PdfReader
from docx import Document
from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI

# Langchainのインポート
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Streamlitのページ設定を最初に記述
st.set_page_config(
    page_title="セミナータイトルジェネレーター（チャット版）",
    layout="wide",
    initial_sidebar_state="expanded"
)

# スタイル設定
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stChatMessage {margin-bottom: 1rem;}
.user-message {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
.assistant-message {background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
.evaluation-card {border: 2px solid #4CAF50; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
.option-button {margin: 0.2rem; padding: 0.5rem 1rem; border-radius: 0.3rem;}
</style>
""", unsafe_allow_html=True)

# データクラスの定義（元のまま使用）
@dataclass
class WebContent:
    title: str
    description: str
    main_content: str
    error: Optional[str] = None

@dataclass
class PainPoint:
    headline: str
    description: str
    selected: bool = False
    
    def to_text(self) -> str:
        return f"{self.headline}：{self.description}"
    
    def combined_text(self) -> str:
        return f"{self.headline}\n\n{self.description}"

@dataclass
class TitleAnalysis:
    predicted_speed: float
    grade: str
    attractive_words: List[str]
    has_specific_problem: bool
    has_exclamation: bool
    title_length: int
    category_score: float
    reasoning: Dict[str, str]
    evaluation_comment: str

@dataclass
class TitleEvaluation:
    speed: float
    grade: str
    comment: str
    timestamp: str = datetime.now().isoformat()

@dataclass
class GeneratedTitle:
    main_title: str
    sub_title: str
    evaluation: TitleEvaluation
    original_main_title: str
    original_sub_title: str

@dataclass
class HeadlineSet:
    background: str
    problem: str
    solution: str

    def to_dict(self):
        return {
            "background": self.background,
            "problem": self.problem,
            "solution": self.solution
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            background=data.get("background", ""),
            problem=data.get("problem", ""),
            solution=data.get("solution", "")
        )

# URLContentExtractorクラス
class URLContentExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0'
        }

    def extract_with_trafilatura(self, url: str) -> Optional[WebContent]:
        try:
            downloaded = fetch_url(url)
            if downloaded is None:
                return WebContent(
                    title="",
                    description="",
                    main_content="",
                    error="URLからのコンテンツ取得に失敗しました"
                )

            content = extract(downloaded, include_comments=False, include_tables=False)
            if content is None:
                return WebContent(
                    title="",
                    description="",
                    main_content="",
                    error="コンテンツの抽出に失敗しました"
                )

            soup = BeautifulSoup(downloaded, 'html.parser')
            title = soup.title.string if soup.title else ""
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ""

            return WebContent(
                title=title,
                description=description,
                main_content=content
            )
        except Exception as e:
            return WebContent(
                title="",
                description="",
                main_content="",
                error=f"エラーが発生しました: {str(e)}"
            )

# Pydanticモデルの定義
class RefinedTitles(BaseModel):
    main_title: str = Field(description="修正後のメインタイトル")
    sub_title: str = Field(description="修正後のサブタイトル")

    def model_dump(self) -> Dict[str, str]:
        return {
            "main_title": self.main_title,
            "sub_title": self.sub_title,
        }

class BodySection(BaseModel):
    refined_text: str = Field(description="修正後のセクションテキスト")

    def model_dump(self) -> Dict[str, str]:
        return {
            "refined_text": self.refined_text,
        }

class RefinedPainPoint(BaseModel):
    headline: str = Field(description="修正後のペインポイント見出し")
    description: str = Field(description="修正後のペインポイント詳細")

    def model_dump(self) -> Dict[str, str]:
        return {
            "headline": self.headline,
            "description": self.description,
        }

# PainPointGeneratorクラス
class PainPointGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.url_extractor = URLContentExtractor()
        self.user_editable_prompt = """
あなたは顧客の課題を深く理解するプロフェッショナルなマーケターです。
提供された情報をもとに、製品・サービスが解決できる主要な顧客の課題（ペインポイント）を抽出してください。

# 指示
- 製品情報と強み・差別化ポイントを分析し、それらが最も効果的に解決できる課題を特定してください
- 対象となるターゲット層が実際に直面している具体的な課題を抽出してください
- 課題は具体的かつ明確に記述し、抽象的な表現は避けてください
- 各ペインポイントは100-150文字程度で簡潔に記述してください
- **ターゲット像を意識して課題を設定してください**

# ターゲット像
{target}

# 強み・差別化ポイント
{strengths}
"""
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください：
{
    "pain_points": [
        {
            "headline": "ペインポイント1の見出し（15-20文字）",
            "description": "ペインポイント1の詳細な説明（100-150文字）"
        },
        {
            "headline": "ペインポイント2の見出し（15-20文字）",
            "description": "ペインポイント2の詳細な説明（100-150文字）"
        },
        {
            "headline": "ペインポイント3の見出し（15-20文字）",
            "description": "ペインポイント3の詳細な説明（100-150文字）"
        }
    ]
}
"""

    def generate_pain_points(self, target: str, strengths: str, prompt_template: str = None, product_urls: List[str] = None, file_contents: List[str] = None) -> List[PainPoint]:
        additional_context = ""
        
        if product_urls:
            for i, url in enumerate(product_urls, 1):
                if url:
                    content = self.url_extractor.extract_with_trafilatura(url)
                    if content and not content.error:
                        additional_context += f"""
製品{i}タイトル: {content.title}
製品{i}説明: {content.description}
製品{i}詳細: {content.main_content[:1000]}
"""

        if file_contents:
            for i, file_content in enumerate(file_contents, 1):
                additional_context += f"""
アップロードされたファイル{i}の内容:
{file_content}
"""

        prompt = (prompt_template or self.user_editable_prompt).format(target=target, strengths=strengths) + additional_context + self.fixed_output_instructions

        result_text = None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なマーケターです。"},
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
                else:
                    raise ValueError("ペインポイントを抽出できませんでした")

            if not isinstance(result, dict) or "pain_points" not in result:
                raise ValueError("不正な応答形式です")

            pain_points = result["pain_points"]
            if not isinstance(pain_points, list) or not pain_points:
                raise ValueError("ペインポイントが見つかりません")

            return [PainPoint(headline=p.get("headline", ""), description=p.get("description", "")) for p in pain_points[:3]]

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            if result_text:
                st.error(f"AIからの応答:\n{result_text}")
            return []

    def refine_pain_point(self, pain_point_text: str, prompt: str) -> Optional[Dict[str, str]]:
        lines = pain_point_text.strip().split('\n\n', 1)
        if len(lines) == 2:
            headline, description = lines
        else:
            headline = pain_point_text[:15] + "..." if len(pain_point_text) > 15 else pain_point_text
            description = pain_point_text
        
        parser = PydanticOutputParser(pydantic_object=RefinedPainPoint)

        prompt_template = PromptTemplate(
            template="以下のペインポイントを修正してください。\n{format_instructions}\n見出し: {headline}\n詳細: {description}\n要望: {prompt}",
            input_variables=["headline", "description", "prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        llm = ChatOpenAI(temperature=0, model=self.model, openai_api_key=self.client.api_key)
        chain = prompt_template | llm | parser

        try:
            output = chain.invoke({"headline": headline, "description": description, "prompt": prompt})
            return output
        except Exception as e:
            st.error(f"Langchainによるペインポイント修正でエラーが発生しました: {e}")
            return None

# TitleGeneratorクラス
class TitleGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.url_extractor = URLContentExtractor()
        self.user_editable_prompt = """
あなたはセミナータイトルの生成を行うプロフェッショナルなコピーライターです。以下の制約条件と入力された情報をもとにセミナータイトルを生成してください。

# 制約条件
- メインタイトルとサブタイトルに分ける
- メインタイトルでは、問題点や課題、悩み、不安を投げかける
- サブタイトルでは、メインタイトルで表現したインサイトを解決する手段や手法、アプローチ、その先に得られるベネフィットを表現する
- メインタイトル、サブタイトルは、それぞれ40文字以内で簡潔にする
- 感嘆符（！）は使用しない
- 参加したら何がわかるのかが明確である
- **ターゲット像を意識する**
- **選択されたペインポイントに基づいたタイトルにする**

# ターゲット像
{target}

# 選択されたペインポイント
【見出し】{pain_point_headline}
【詳細】{pain_point_description}

# Steps

1. 入力情報、ターゲット像、ペインポイントから製品の特徴とターゲットのペインポイントを理解する
2. ターゲットのペインポイントに基づき、メインタイトルで問題点や課題、悩み、不安を投げかける
3. 製品の特徴に基づき、サブタイトルでメインタイトルで表現したインサイトを解決する手段や手法、アプローチ、その先に得られるベネフィットを表現する
4. メインタイトルとサブタイトルをそれぞれ40文字以内で簡潔に表現する
5. メインタイトルとサブタイトルを組み合わせ、ターゲットが参加したら何がわかるのかが明確なタイトルを生成する
6. 感嘆符（！）が使用されていないことを確認する

# Examples

- **Main Title**: 人材不足でも、社内ネットワークを安定稼働し続けるにはどうすればよいのか？
- **Subtitle**: 〜ネットワーク障害解決を迅速化するマップ機能の活用法を解説〜
"""
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

    def generate_titles(self, context: str, target: str, pain_point: PainPoint, prompt_template: str = None, product_urls: List[str] = None, file_contents: List[str] = None) -> List[Dict[str, str]]:
            additional_context = ""
            
            if product_urls:
                for i, url in enumerate(product_urls, 1):
                    if url:
                        content = self.url_extractor.extract_with_trafilatura(url)
                        if content and not content.error:
                            additional_context += f"""
製品{i}タイトル: {content.title}
製品{i}説明: {content.description}
製品{i}詳細: {content.main_content[:1000]}
"""

            if file_contents:
                for i, file_content in enumerate(file_contents, 1):
                    additional_context += f"""
アップロードされたファイル{i}の内容:
{file_content}
"""

            prompt = f"""
# 入力情報
{context}
""" + (prompt_template or self.user_editable_prompt).format(
                target=target, 
                pain_point_headline=pain_point.headline,
                pain_point_description=pain_point.description
            ) + additional_context + self.fixed_output_instructions

            result_text = None

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
                    else:
                        raise ValueError("タイトルを抽出できませんでした")

                if not isinstance(result, dict) or "titles" not in result:
                    raise ValueError("不正な応答形式です")

                titles = result["titles"]
                if not isinstance(titles, list) or not titles:
                    raise ValueError("タイトルが見つかりません")

                return titles[:3]

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                if result_text:
                    st.error(f"AIからの応答:\n{result_text}")
                return []

    def refine_title(self, main_title: str, sub_title: str, prompt: str) -> Optional[Dict[str, str]]:
        parser = PydanticOutputParser(pydantic_object=RefinedTitles)

        prompt_template = PromptTemplate(
            template="以下のメインタイトルとサブタイトルを修正してください。\n{format_instructions}\nメインタイトル: {main_title}\nサブタイトル: {sub_title}\n要望: {prompt}",
            input_variables=["main_title", "sub_title", "prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        llm = ChatOpenAI(temperature=0, model=self.model, openai_api_key=self.client.api_key)
        chain = prompt_template | llm | parser

        try:
            output = chain.invoke({"main_title": main_title, "sub_title": sub_title, "prompt": prompt})
            return output
        except Exception as e:
            st.error(f"Langchainによるタイトル修正でエラーが発生しました: {e}")
            return None

# HeadlineGeneratorクラス
class HeadlineGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fixed_prompt_part = """
「『{title}』というタイトルのイベントを企画しており、その告知文を作成します。 告知文を作成する前に、以下の内容でその見出しを３つ作成してください。それぞれの見出しは簡潔な文章としてください。
# 出力形式
- 見出しはマークダウン形式（#記号）を使わず、プレーンテキストとして出力してください。
- それぞれの見出しは30文字程度の簡潔な文章としてください。
」
"""
        self.user_editable_prompt = """
見出し1：このセミナーを開催する、社会や企業の背景
見出し2：このセミナーで訴求したい、課題、問題、悩み、不安
見出し3：上記課題の解決の方向性
- **ターゲット像を意識する**

# ターゲット像
{target}

# 選択されたペインポイント
【見出し】{pain_point_headline}
【詳細】{pain_point_description}
"""
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください：
{
    "background": "背景の見出し",
    "problem": "課題の見出し",
    "solution": "解決策の見出し"
}
"""

    def generate_headlines(self, title: str, target: str, pain_point: PainPoint, prompt_template: str = None) -> HeadlineSet:
            prompt = self.fixed_prompt_part.format(title=title) + (prompt_template or self.user_editable_prompt).format(
                target=target, 
                pain_point_headline=pain_point.headline,
                pain_point_description=pain_point.description
            ) + self.fixed_output_instructions

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                        {"role": "user", "content": "あなたは優秀なコピーライターです。ターゲット像を意識して見出しを作成してください。"},
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
                st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
                return HeadlineSet("", "", "")

# BodyGeneratorクラス
class BodyGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fixed_prompt_part = """
以下のセミナータイトルと見出しに基づいて、本文を生成してください。
**重要**: 各見出しはマークダウン形式で出力してください。
見出しは必ず以下の形式で、1行ずつ記述してください:

## {background}

## {problem}

## {solution}

見出しは変更せず、そのまま使用してください。各セクションの本文は見出しの下に記述してください。
"""
        self.user_editable_prompt = """
以下の制約条件と入力情報を踏まえて本文を生成してください。

# 制約条件
- 各見出しセクションは最低300文字以上とし、3文以内でまとめてください（句読点で区切られた3文以内）。
- 全文で1000文字以内に収めてください。
- 本文中では箇条書きを使用しないでください。
- 3つの見出しを通して、一連のストーリーとして流れを持たせてください。
- セミナー内容の紹介および参加を促す表現は、3つ目の見出しのセクションでのみ行ってください。
- 3つ目の見出しのセクションは「本セミナーでは、」から始めてください。
- 重要なキーワードは本文中に必ず含めてください。
- あくまでセミナー集客用の文章であることを念頭に、魅力的かつ説得力のある内容にしてください。
- **見出し１（背景）と見出し２（課題）の本文では、解決策には触れないでください。**
- **ターゲット像を意識する**
- **選択されたペインポイントに沿った内容にする**

# ターゲット像
{target}

# 選択されたペインポイント
【見出し】{pain_point_headline}
【詳細】{pain_point_description}
"""

    def generate_body(self, title: str, headlines: HeadlineSet, target: str, pain_point: PainPoint, prompt_template: str = None) -> str:
        prompt = self.fixed_prompt_part.format(
            background=headlines.background,
            problem=headlines.problem,
            solution=headlines.solution
        ) + (prompt_template or self.user_editable_prompt).format(
            target=target, 
            pain_point_headline=pain_point.headline,
            pain_point_description=pain_point.description
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": "あなたは優秀なコピーライターです。ターゲット像を意識して本文を作成してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return ""

    def refine_body_section(self, section_text: str, prompt: str, section_type: str) -> Optional[Dict[str, str]]:
        parser = PydanticOutputParser(pydantic_object=BodySection)

        prompt_template = PromptTemplate(
            template=f"以下の{section_type}セクションを修正してください。\n{{format_instructions}}\n{section_type}セクション:\n{{section_text}}\n要望: {{prompt}}",
            input_variables=["section_text", "prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        llm = ChatOpenAI(temperature=0, model=self.model, openai_api_key=self.client.api_key)
        chain = prompt_template | llm | parser

        try:
            output = chain.invoke({"section_text": section_text, "prompt": prompt})
            return output
        except Exception as e:
            st.error(f"Langchainによる本文セクション修正でエラーが発生しました: {e}")
            return None

    def parse_body_sections(self, body_text, headlines):
        """見出しと本文を安定的に分離するパーサー"""
        background_pattern = re.escape(f"## {headlines.background}")
        problem_pattern = re.escape(f"## {headlines.problem}")
        solution_pattern = re.escape(f"## {headlines.solution}")
        
        background_match = re.search(background_pattern, body_text)
        problem_match = re.search(problem_pattern, body_text)
        solution_match = re.search(solution_pattern, body_text)
        
        if not all([background_match, problem_match, solution_match]):
            sections = re.split(r'##\s+', body_text)
            if len(sections) >= 4:
                sections = sections[1:]
                return {
                    "background": sections[0].strip(),
                    "problem": sections[1].strip() if len(sections) > 1 else "",
                    "solution": sections[2].strip() if len(sections) > 2 else ""
                }
            return None
        
        background_text = body_text[background_match.end():problem_match.start()].strip()
        problem_text = body_text[problem_match.end():solution_match.start()].strip()
        solution_text = body_text[solution_match.end():].strip()
        
        return {
            "background": background_text,
            "problem": problem_text,
            "solution": solution_text
        }

# SeminarTitleEvaluatorクラス
class SeminarTitleEvaluator:
    def __init__(self, seminar_data: pd.DataFrame):
        self.df = seminar_data
        self._initialize_analytics()

    def _initialize_analytics(self):
        high_performing = self.df[self.df['Acquisition_Speed'] >= 2.5]
        self.attractive_words = self._extract_effective_keywords(high_performing)
        self.problem_indicators = [
            '課題', '問題', 'による', 'ための', '向上', '改善', '解決', '対策',
            'どうする', 'なぜ', 'どう', '方法', '実現', 'ポイント', '実践',
            'ベストプラクティス', 'ノウハウ', '事例', '成功'
        ]

    def _extract_effective_keywords(self, high_performing_df) -> List[str]:
        words = []
        for title in high_performing_df['Seminar_Title']:
            if isinstance(title, str):
                clean_title = (title.replace('〜', ' ')
                                 .replace('、', ' ')
                                 .replace('【', ' ')
                                 .replace('】', ' ')
                                 .replace('「', ' ')
                                 .replace('」', ' '))
                title_words = [w for w in clean_title.split()
                              if len(w) > 1 and not w.isdigit()]
                words.extend(title_words)

        word_counts = pd.Series(words).value_counts()
        return list(word_counts[word_counts >= 2].index)

    def _generate_evaluation_comment(self, analysis_data: dict) -> str:
        comments = []

        if analysis_data["predicted_speed"] >= 2.5:
            comments.append("高い集客が期待できます")
        elif analysis_data["predicted_speed"] >= 1.8:
            comments.append("一定の集客が見込めます")
        else:
            comments.append("改善の余地があります")

        if analysis_data["attractive_words"]:
            comments.append("効果的なキーワードが含まれています")
        else:
            comments.append("効果的なキーワードの追加を検討してください")

        if analysis_data["title_length"] > 40:
            comments.append("タイトルを短くすることを推奨します")

        if not analysis_data["has_specific_problem"]:
            comments.append("具体的な課題や問題提起の追加を検討してください")

        return "。".join(comments)

    def evaluate_title(self, title: str) -> TitleAnalysis:
        base_score = self._calculate_base_score(title)
        final_score = min(max(base_score, 1.0), 3.0)

        matching_words = [word for word in self.attractive_words if word in title]
        has_problem = any(indicator in title for indicator in self.problem_indicators)

        reasoning = {
            "keywords": f"効果的なキーワード: {', '.join(matching_words) if matching_words else '該当なし'}",
            "title_length": f"タイトルの長さ: {len(title)}文字 （{'適切' if len(title) <= 40 else '長い'}）",
            "problem_indication": f"問題提起: {'あり' if has_problem else 'なし'}",
            "exclamation": f"感嘆符: {'あり（減点）' if '!' in title or '！' in title else 'なし'}",
            "predicted_speed": f"予測される集客速度: {final_score:.1f}"
        }

        grade = 'A' if final_score >= 2.5 else 'B' if final_score >= 1.8 else 'C'

        analysis_data = {
            "predicted_speed": final_score,
            "attractive_words": matching_words,
            "has_specific_problem": has_problem,
            "title_length": len(title)
        }

        evaluation_comment = self._generate_evaluation_comment(analysis_data)

        return TitleAnalysis(
            predicted_speed=final_score,
            grade=grade,
            attractive_words=matching_words,
            has_specific_problem=has_problem,
            has_exclamation='!' in title or '！' in title,
            title_length=len(title),
            category_score=0.0,
            reasoning=reasoning,
            evaluation_comment=evaluation_comment
        )

    def _calculate_base_score(self, title: str) -> float:
        base_score = 1.0

        matching_words = [word for word in self.attractive_words if word in title]
        keyword_score = len(matching_words) * 0.4
        base_score += min(keyword_score, 1.2)

        title_length = len(title)
        if title_length <= 20:
            base_score += 0.3
        elif title_length <= 40:
            base_score += 0.1
        elif title_length > 60:
            base_score -= 0.2

        if any(indicator in title for indicator in self.problem_indicators):
            base_score += 0.4

        if '!' in title or '！' in title:
            base_score -= 0.3

        return base_score

# InMemoryCacheクラス
class InMemoryCache:
    def __init__(self):
        if 'title_cache' not in st.session_state:
            st.session_state.title_cache = {}

    def get_evaluation(self, title: str) -> Optional[TitleEvaluation]:
        return st.session_state.title_cache.get(title)

    def set_evaluation(self, title: str, evaluation: TitleEvaluation):
        st.session_state.title_cache[title] = evaluation

# データ読み込み関数
def init_bigquery_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials)

def load_seminar_data():
    client = init_bigquery_client()

    query = """
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

    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"データの読み込みでエラーが発生しました: {str(e)}")
        return None

# フォーマット生成関数
def generate_mid_review_format(開催日, 主催企業, 集客人数, 初稿UP期限, 参考情報, ターゲット, 強み, pain_point, seminar_title, headlines):
    format_text = f"""【ペインポイント・タイトル・見出し案の確認依頼】

下記、ご確認をお願いします。

＜対象セミナー＞
・開催日：{開催日}
・主催企業：{主催企業}
・集客人数：{集客人数}
・初稿UP期限：{初稿UP期限}

＜商材＞
{参考情報}

＜ターゲット＞
{ターゲット}

＜強み・差別化ポイント＞
{強み}

＜ペインポイント＞
{pain_point.headline}
{pain_point.description}

＜セミナータイトル案＞
{seminar_title}

＜見出し＞
# {headlines.background}
# {headlines.problem}
# {headlines.solution}
"""
    return format_text

def generate_plan_review_format(開催日, 主催企業, 集客人数, 初稿UP期限, 参考情報, セミナータイトル, 
                               見出し_background, 見出し_problem, 見出し_solution, 
                               ターゲット, pain_point, 強み,
                               background_text, problem_text, solution_text):
    format_text = f"""【タイトル・見出し・本文の確認依頼】

下記、ご確認をお願いします。

＜対象セミナー＞
・開催日：{開催日}
・主催企業：{主催企業}
・集客人数：{集客人数}
・初稿UP期限：{初稿UP期限}

＜商材＞
{参考情報}

＜ターゲット＞
{ターゲット}

＜強み・差別化ポイント＞
{強み}

＜ペインポイント＞
{pain_point.headline}
{pain_point.description}

＜オファー＞
※ここに追記※

＜告知文＞
■セミナータイトル：
{セミナータイトル}

■見出し・本文：
# {見出し_background}
{background_text}

# {見出し_problem}
{problem_text}

# {見出し_solution}
{solution_text}
"""
    return format_text

def parse_pain_point_input(pain_point_text: str) -> PainPoint:
    lines = pain_point_text.strip().split("\n", 1)
    
    if len(lines) >= 2:
        headline = lines[0].strip()
        description = lines[1].strip()
    else:
        text = lines[0].strip()
        headline = text[:30] + ("..." if len(text) > 30 else "")
        description = text
    
    return PainPoint(headline=headline, description=description)

def parse_mid_review_format(mid_review_text: str) -> Optional[Dict[str, str]]:
    try:
        lines = mid_review_text.strip().split("\n")
        result = {}
        
        current_section = None
        pain_point_headline = ""
        pain_point_description = ""
        reading_seminar_info = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "＜対象セミナー＞" in line:
                reading_seminar_info = True
                continue
            
            if reading_seminar_info and line.startswith("・"):
                if "開催日：" in line:
                    result["seminar_開催日"] = line.split("開催日：")[1].strip()
                elif "主催企業：" in line:
                    result["seminar_主催企業"] = line.split("主催企業：")[1].strip()
                elif "集客人数：" in line:
                    result["seminar_集客人数"] = line.split("集客人数：")[1].strip()
                elif "初稿UP期限：" in line:
                    result["seminar_初稿UP期限"] = line.split("初稿UP期限：")[1].strip()
                
            if "＜ターゲット＞" in line:
                reading_seminar_info = False
                current_section = "target"
                continue
            elif "＜強み・差別化ポイント＞" in line:
                current_section = "strengths"
                continue
            elif "＜ペインポイント＞" in line:
                current_section = "pain_point"
                continue
            elif "＜セミナータイトル案＞" in line:
                current_section = "title"
                continue
            elif "＜見出し＞" in line:
                current_section = "headline"
                continue
                
            if current_section == "target":
                if "target" not in result:
                    result["target"] = line
                else:
                    result["target"] += "\n" + line
            elif current_section == "strengths":
                if "strengths" not in result:
                    result["strengths"] = line
                else:
                    result["strengths"] += "\n" + line
            elif current_section == "pain_point":
                if not pain_point_headline:
                    pain_point_headline = line
                elif not pain_point_description:
                    pain_point_description = line
                else:
                    pain_point_description += "\n" + line
            elif current_section == "title":
                result["seminar_title"] = line
            elif current_section == "headline":
                if line.startswith("# "):
                    line = line[2:].strip()
                    if "headline_background" not in result:
                        result["headline_background"] = line
                    elif "headline_problem" not in result:
                        result["headline_problem"] = line
                    elif "headline_solution" not in result:
                        result["headline_solution"] = line
        
        result["pain_point_headline"] = pain_point_headline
        result["pain_point_description"] = pain_point_description
        
        return result
    except Exception as e:
        st.error(f"中間レビューの解析中にエラーが発生しました: {e}")
        return None

# チャットアシスタントクラス
class ChatAssistant:
    def __init__(self, api_key: str, evaluator: SeminarTitleEvaluator):
        self.pain_point_generator = PainPointGenerator(api_key)
        self.title_generator = TitleGenerator(api_key)
        self.headline_generator = HeadlineGenerator(api_key)
        self.body_generator = BodyGenerator(api_key)
        self.evaluator = evaluator
        self.cache = InMemoryCache()
        self.url_extractor = URLContentExtractor()
        
    def parse_user_message(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """ユーザーメッセージを解析してアクションを決定"""
        message_lower = message.lower()
        
        # アクションパターンの定義
        patterns = {
            "start": ["始め", "スタート", "開始", "こんにちは", "hello"],
            "pain_point": ["ペインポイント", "課題", "痛み", "問題点"],
            "title": ["タイトル", "題名", "見出し案"],
            "headline": ["見出し", "章立て", "構成"],
            "body": ["本文", "文章", "内容", "コンテンツ"],
            "refine": ["修正", "変更", "改善", "直し"],
            "evaluate": ["評価", "採点", "レビュー"],
            "format": ["フォーマット", "slack", "投稿"],
            "url": ["url", "リンク", "ページ"],
            "file": ["ファイル", "アップロード", "添付"],
            "resume": ["再開", "続き", "復帰"],
        }
        
        for action, keywords in patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return action, {}
                
        return "general", {}
    
    def generate_response(self, user_message: str, context: Dict[str, Any]) -> str:
        """ユーザーメッセージに対する応答を生成"""
        action, params = self.parse_user_message(user_message)
        
        if action == "start":
            return self.handle_start()
        elif action == "pain_point":
            return self.handle_pain_point_request(context)
        elif action == "title":
            return self.handle_title_request(context)
        elif action == "headline":
            return self.handle_headline_request(context)
        elif action == "body":
            return self.handle_body_request(context)
        elif action == "refine":
            return self.handle_refine_request(user_message, context)
        elif action == "evaluate":
            return self.handle_evaluate_request(user_message, context)
        elif action == "format":
            return self.handle_format_request(context)
        elif action == "resume":
            return self.handle_resume_request(user_message)
        else:
            return self.handle_general_request(user_message, context)
    
    def handle_start(self) -> str:
        return """こんにちは！セミナータイトル・告知文ジェネレーターへようこそ。

以下の流れでサポートいたします：
1. 基本情報の入力（ターゲット層、強み・差別化ポイント）
2. ペインポイントの生成
3. タイトルの生成と評価
4. 見出しの生成
5. 本文の生成
6. Slack投稿フォーマットの作成

まずは、**ターゲット層**と**強み・差別化ポイント**を教えてください。
製品URLやファイルがあれば、それも共有してください。"""
    
    def handle_pain_point_request(self, context: Dict[str, Any]) -> str:
        if not context.get('target_audience') or not context.get('strengths'):
            return "ペインポイントを生成するために、まずターゲット層と強み・差別化ポイントを教えてください。"
        
        # ペインポイント生成
        pain_points = self.pain_point_generator.generate_pain_points(
            context['target_audience'],
            context['strengths'],
            product_urls=context.get('product_urls', []),
            file_contents=context.get('file_contents', [])
        )
        
        if pain_points:
            response = "以下のペインポイントを生成しました：\n\n"
            for i, pp in enumerate(pain_points, 1):
                response += f"**{i}. {pp.headline}**\n{pp.description}\n\n"
            response += "どのペインポイントを選択しますか？番号で教えてください。\nまた、修正したい場合は「1番を〜に修正」のように指示してください。"
            return response
        else:
            return "ペインポイントの生成に失敗しました。もう一度お試しください。"
    
    def handle_title_request(self, context: Dict[str, Any]) -> str:
        if not context.get('selected_pain_point'):
            return "まずペインポイントを選択してください。"
        
        # タイトル生成
        titles = self.title_generator.generate_titles(
            f"強み・差別化ポイント: {context['strengths']}",
            context['target_audience'],
            context['selected_pain_point'],
            product_urls=context.get('product_urls', []),
            file_contents=context.get('file_contents', [])
        )
        
        if titles:
            response = "以下のタイトル案を生成しました：\n\n"
            for i, title in enumerate(titles, 1):
                full_title = f"{title['main_title']} - {title['sub_title']}"
                analysis = self.evaluator.evaluate_title(full_title)
                response += f"**案{i}**\n"
                response += f"メイン: {title['main_title']}\n"
                response += f"サブ: {title['sub_title']}\n"
                response += f"評価: **{analysis.grade}** (速度: {analysis.predicted_speed:.1f})\n"
                response += f"コメント: {analysis.evaluation_comment}\n\n"
            response += "どのタイトルを選択しますか？番号で教えてください。"
            return response
        else:
            return "タイトルの生成に失敗しました。もう一度お試しください。"
    
    def handle_headline_request(self, context: Dict[str, Any]) -> str:
        if not context.get('selected_title'):
            return "まずタイトルを選択してください。"
        
        # 見出し生成
        headlines = self.headline_generator.generate_headlines(
            context['selected_title'],
            context['target_audience'],
            context['selected_pain_point']
        )
        
        if headlines:
            response = "以下の見出しを生成しました：\n\n"
            response += f"**背景**: {headlines.background}\n"
            response += f"**課題**: {headlines.problem}\n"
            response += f"**解決策**: {headlines.solution}\n\n"
            response += "これらの見出しで本文を生成しますか？修正が必要な場合はお知らせください。"
            return response
        else:
            return "見出しの生成に失敗しました。もう一度お試しください。"
    
    def handle_body_request(self, context: Dict[str, Any]) -> str:
        if not context.get('headlines'):
            return "まず見出しを生成してください。"
        
        # 本文生成
        body = self.body_generator.generate_body(
            context['selected_title'],
            context['headlines'],
            context['target_audience'],
            context['selected_pain_point']
        )
        
        if body:
            response = "以下の本文を生成しました：\n\n"
            response += body + "\n\n"
            response += "修正が必要な箇所があれば、「背景セクションを〜に修正」のように指示してください。"
            return response
        else:
            return "本文の生成に失敗しました。もう一度お試しください。"
    
    def handle_refine_request(self, user_message: str, context: Dict[str, Any]) -> str:
        # 修正対象を特定
        if "ペインポイント" in user_message:
            # ペインポイントの修正処理
            match = re.search(r'(\d+)番?を(.+)に修正', user_message)
            if match:
                index = int(match.group(1)) - 1
                instruction = match.group(2)
                if 0 <= index < len(context.get('pain_points', [])):
                    pain_point = context['pain_points'][index]
                    refined = self.pain_point_generator.refine_pain_point(
                        pain_point.combined_text(),
                        instruction
                    )
                    if refined:
                        return f"ペインポイント{index+1}を修正しました：\n\n**{refined['headline']}**\n{refined['description']}"
        
        elif "タイトル" in user_message:
            # タイトルの修正処理
            return "タイトルの修正指示を受け付けました。どのように修正しますか？"
        
        elif "見出し" in user_message:
            # 見出しの修正処理
            return "見出しの修正指示を受け付けました。どの見出しをどのように修正しますか？"
        
        elif "本文" in user_message or "セクション" in user_message:
            # 本文の修正処理
            return "本文の修正指示を受け付けました。どのセクションをどのように修正しますか？"
        
        return "何を修正したいか、もう少し詳しく教えてください。"
    
    def handle_evaluate_request(self, user_message: str, context: Dict[str, Any]) -> str:
        # タイトル評価の処理
        # メッセージからタイトルを抽出
        lines = user_message.split('\n')
        main_title = ""
        sub_title = ""
        
        for line in lines:
            if "メイン" in line:
                main_title = line.split("：")[-1].strip() if "：" in line else line.split(":")[-1].strip()
            elif "サブ" in line:
                sub_title = line.split("：")[-1].strip() if "：" in line else line.split(":")[-1].strip()
        
        if main_title:
            full_title = f"{main_title} - {sub_title}" if sub_title else main_title
            analysis = self.evaluator.evaluate_title(full_title)
            
            response = f"タイトル評価結果：\n\n"
            response += f"**タイトル**: {full_title}\n"
            response += f"**評価**: {analysis.grade} (速度: {analysis.predicted_speed:.1f})\n"
            response += f"**コメント**: {analysis.evaluation_comment}\n\n"
            
            for reason in analysis.reasoning.values():
                response += f"- {reason}\n"
            
            return response
        else:
            return "評価したいタイトルを「メイン：〜」「サブ：〜」の形式で教えてください。"
    
    def handle_format_request(self, context: Dict[str, Any]) -> str:
        if not all([context.get('selected_title'), context.get('headlines'), context.get('body')]):
            return "Slack投稿フォーマットを生成するには、タイトル、見出し、本文が必要です。"
        
        # 必要な情報を収集
        return """Slack投稿フォーマットを生成します。以下の情報を教えてください：
- 開催日
- 主催企業
- 集客人数
- 初稿UP期限

例：「開催日：3/15、主催企業：株式会社〇〇、集客人数：100名、初稿UP期限：3/10」"""
    
    def handle_resume_request(self, user_message: str) -> str:
        if "中間レビュー" in user_message:
            return """中間レビューフォーマットを貼り付けてください。
フォーマットから情報を読み取って、本文生成から再開します。"""
        else:
            return """どこから再開しますか？
- ペインポイントから再開する場合：「ペインポイントから再開」
- 中間レビューから再開する場合：「中間レビューから再開」"""
    
    def handle_general_request(self, user_message: str, context: Dict[str, Any]) -> str:
        # 一般的な質問への対応
        return """申し訳ございません。ご質問の内容がよく理解できませんでした。

以下のようなリクエストに対応できます：
- 「ペインポイントを生成」
- 「タイトルを生成」
- 「見出しを生成」
- 「本文を生成」
- 「〜を修正」
- 「タイトルを評価」
- 「Slackフォーマットを生成」

どのようなサポートが必要でしょうか？"""

def init_session_state():
    """セッション状態の初期化"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'context' not in st.session_state:
        st.session_state.context = {
            'target_audience': '',
            'strengths': '',
            'product_urls': [],
            'file_contents': [],
            'pain_points': [],
            'selected_pain_point': None,
            'generated_titles': [],
            'selected_title': None,
            'headlines': None,
            'body': None,
            'body_sections': {},
        }
    if 'seminar_data' not in st.session_state:
        st.session_state.seminar_data = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'start'

def display_chat_history():
    """チャット履歴の表示"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "components" in message:
                # 特殊なコンポーネントの表示
                display_components(message["components"])
            else:
                st.markdown(message["content"])

def display_components(components):
    """特殊なコンポーネント（ボタン、選択肢など）の表示"""
    for comp in components:
        if comp["type"] == "options":
            for option in comp["options"]:
                if st.button(option["label"], key=f"{comp['id']}_{option['value']}"):
                    handle_option_click(comp["id"], option["value"])
        elif comp["type"] == "input":
            value = st.text_input(comp["label"], key=comp["id"])
            if st.button("送信", key=f"{comp['id']}_submit"):
                handle_input_submit(comp["id"], value)

def handle_option_click(component_id: str, value: str):
    """オプションクリックの処理"""
    if component_id == "pain_point_select":
        st.session_state.context['selected_pain_point'] = st.session_state.context['pain_points'][int(value)]
        st.session_state.messages.append({
            "role": "user",
            "content": f"ペインポイント{int(value)+1}を選択しました"
        })
        st.rerun()
    elif component_id == "title_select":
        st.session_state.context['selected_title'] = value
        st.session_state.messages.append({
            "role": "user",
            "content": f"タイトル案を選択しました"
        })
        st.rerun()

def handle_input_submit(component_id: str, value: str):
    """入力送信の処理"""
    if component_id == "seminar_info":
        # セミナー情報の処理
        pass

def main():
    init_session_state()
    
    # APIキーの確認
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OpenAI APIキーが設定されていません")
        return
    
    st.title("💬 セミナータイトル・告知文ジェネレーター")
    
    # サイドバー
    with st.sidebar:
        st.header("📋 現在の情報")
        
        if st.session_state.context.get('target_audience'):
            st.write("**ターゲット層:**")
            st.info(st.session_state.context['target_audience'])
        
        if st.session_state.context.get('strengths'):
            st.write("**強み・差別化ポイント:**")
            st.info(st.session_state.context['strengths'])
        
        if st.session_state.context.get('selected_pain_point'):
            st.write("**選択されたペインポイント:**")
            pp = st.session_state.context['selected_pain_point']
            st.info(f"{pp.headline}\n{pp.description}")
        
        if st.session_state.context.get('selected_title'):
            st.write("**選択されたタイトル:**")
            st.info(st.session_state.context['selected_title'])
        
        st.divider()
        
        # ファイルアップロード
        st.header("📎 ファイルアップロード")
        uploaded_files = st.file_uploader(
            "ファイルを選択",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            file_contents = []
            for file in uploaded_files[:5]:
                try:
                    if file.type == "text/plain":
                        content = file.getvalue().decode('utf-8')
                    elif file.type == "application/pdf":
                        reader = PdfReader(file)
                        content = "\n".join([page.extract_text() for page in reader.pages])
                    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        document = Document(file)
                        content = "\n".join([para.text for para in document.paragraphs])
                    else:
                        continue
                    
                    file_contents.append(content)
                    st.success(f"✅ {file.name}")
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
            
            st.session_state.context['file_contents'] = file_contents
        
        # URLリンク
        st.header("🔗 製品URL")
        for i in range(3):
            url = st.text_input(f"URL {i+1}", key=f"sidebar_url_{i}")
            if url and i < len(st.session_state.context.get('product_urls', [])):
                st.session_state.context['product_urls'][i] = url
    
    # データ読み込み
    if st.session_state.seminar_data is None:
        with st.spinner("データを読み込んでいます..."):
            df = load_seminar_data()
            if df is not None:
                st.session_state.seminar_data = df
                st.session_state.evaluator = SeminarTitleEvaluator(df)
                st.session_state.assistant = ChatAssistant(api_key, st.session_state.evaluator)
    
    # 初回メッセージ
    if not st.session_state.messages:
        welcome_message = st.session_state.assistant.handle_start()
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_message
        })
    
    # チャット履歴の表示
    display_chat_history()
    
    # ユーザー入力
    if prompt := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # アシスタントの応答
        with st.chat_message("assistant"):
            with st.spinner("考えています..."):
                # コンテキストの更新（基本情報の抽出）
                if "ターゲット" in prompt and "強み" in prompt:
                    # ターゲットと強みを抽出
                    lines = prompt.split('\n')
                    for line in lines:
                        if "ターゲット" in line:
                            st.session_state.context['target_audience'] = line.split('：')[-1].strip()
                        elif "強み" in line:
                            st.session_state.context['strengths'] = line.split('：')[-1].strip()
                
                # アシスタントの応答生成
                response = st.session_state.assistant.generate_response(
                    prompt,
                    st.session_state.context
                )
                
                st.markdown(response)
                
                # 応答をメッセージ履歴に追加
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

if __name__ == "__main__":
    main()
