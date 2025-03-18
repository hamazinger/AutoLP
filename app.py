import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
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
    page_title="セミナータイトルジェネレーター",
    layout="wide"
)

# 右下の開発者プロフィールリンクやフッター非表示用CSS
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
/* ペインポイント選択のスタイル */
.selected-pain-point {
    border: 2px solid #4CAF50 !important;
    background-color: #f0f9f0 !important;
    padding: 10px;
    border-radius: 5px;
}
.unselected-pain-point {
    border: 1px solid #e6e6e6;
    padding: 10px;
    border-radius: 5px;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# データクラスの定義
@dataclass
class WebContent:
    title: str
    description: str
    main_content: str
    error: Optional[str] = None

@dataclass
class PainPoint:
    headline: str  # タイトルを見出しに変更
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
                if url:  # URLが空でない場合のみ処理
                    content = self.url_extractor.extract_with_trafilatura(url)
                    if content and not content.error:
                        additional_context += f"""
製品{i}タイトル: {content.title}
製品{i}説明: {content.description}
製品{i}詳細: {content.main_content[:1000]}
"""
                    else:
                        st.warning(f"製品{i}情報の取得に失敗しました: {content.error if content else '不明なエラー'}")

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
        # テキスト全体から見出しと詳細を抽出するロジック
        lines = pain_point_text.strip().split('\n\n', 1)
        if len(lines) == 2:
            headline, description = lines
        else:
            # 区切りがない場合は全体を詳細とし、見出しは最初の15文字を使用
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
                    if url:  # URLが空でない場合のみ処理
                        content = self.url_extractor.extract_with_trafilatura(url)
                        if content and not content.error:
                            additional_context += f"""
製品{i}タイトル: {content.title}
製品{i}説明: {content.description}
製品{i}詳細: {content.main_content[:1000]}
"""
                        else:
                            st.warning(f"製品{i}情報の取得に失敗しました: {content.error if content else '不明なエラー'}")

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

class BodyGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # 本文生成プロンプトをより明確に
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
        # 各見出しをパターンとして準備
        background_pattern = re.escape(f"## {headlines.background}")
        problem_pattern = re.escape(f"## {headlines.problem}")
        solution_pattern = re.escape(f"## {headlines.solution}")
        
        # 各セクションの開始位置を特定
        background_match = re.search(background_pattern, body_text)
        problem_match = re.search(problem_pattern, body_text)
        solution_match = re.search(solution_pattern, body_text)
        
        if not all([background_match, problem_match, solution_match]):
            st.warning("本文のパースに問題が発生しました。正規表現パターンが一致しませんでした。")
            # 代替パターンの試行（単純な ## で始まる行を探す）
            sections = re.split(r'##\s+', body_text)
            if len(sections) >= 4:  # 最初の分割は空になるはず
                sections = sections[1:]  # 最初の空セクションを削除
                return {
                    "background": sections[0].strip(),
                    "problem": sections[1].strip() if len(sections) > 1 else "",
                    "solution": sections[2].strip() if len(sections) > 2 else ""
                }
            return None
        
        # 各セクションのテキストを抽出
        background_text = body_text[background_match.end():problem_match.start()].strip()
        problem_text = body_text[problem_match.end():solution_match.start()].strip()
        solution_text = body_text[solution_match.end():].strip()
        
        return {
            "background": background_text,
            "problem": problem_text,
            "solution": solution_text
        }


# 中間レビュー用フォーマット生成 (見出しまでの情報を含む)
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

# 企画案レビュー用フォーマット生成機能
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
            category_score=0.0,  # カテゴリ評価を使用しないため0.0を固定
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

class InMemoryCache:
    def __init__(self):
        if 'title_cache' not in st.session_state:
            st.session_state.title_cache = {}

    def get_evaluation(self, title: str) -> Optional[TitleEvaluation]:
        return st.session_state.title_cache.get(title)

    def set_evaluation(self, title: str, evaluation: TitleEvaluation):
        st.session_state.title_cache[title] = evaluation

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

def display_evaluation_details(title: str, evaluator: SeminarTitleEvaluator):
    analysis = evaluator.evaluate_title(title)

    st.write("### 評価詳細")
    st.info(f"**評価コメント:** {analysis.evaluation_comment}")

    for reason in analysis.reasoning.values():
        st.write(f"- {reason}")

    if analysis.attractive_words:
        st.write("### タイトル中の効果的なキーワード")
        highlighted_title = title
        for word in analysis.attractive_words:
            highlighted_title = highlighted_title.replace(
                word,
                f'<span style="background-color: #FFEB3B">{word}</span>'
            )
        st.markdown(f'<p>{highlighted_title}</p>', unsafe_allow_html=True)

# 手動入力ペインポイントの解析
def parse_pain_point_input(pain_point_text: str) -> PainPoint:
    # 入力テキストから見出しと詳細を抽出
    lines = pain_point_text.strip().split("\n", 1)
    
    if len(lines) >= 2:
        headline = lines[0].strip()
        description = lines[1].strip()
    else:
        # 区切りがない場合は最初の30文字を見出しとして使用
        text = lines[0].strip()
        headline = text[:30] + ("..." if len(text) > 30 else "")
        description = text
    
    return PainPoint(headline=headline, description=description)

# 中間レビューフォーマットを解析する関数
def parse_mid_review_format(mid_review_text: str) -> Optional[Dict[str, str]]:
    """中間レビューフォーマットから必要な情報を抽出する関数"""
    try:
        lines = mid_review_text.strip().split("\n")
        result = {}
        
        # セクション検出フラグ
        current_section = None
        
        # 一時変数
        pain_point_headline = ""
        pain_point_description = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # セクション判定
            if "＜ターゲット＞" in line:
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
                
            # セクション内容の処理
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
        
        # ペインポイント情報を設定
        result["pain_point_headline"] = pain_point_headline
        result["pain_point_description"] = pain_point_description
        
        return result
    except Exception as e:
        st.error(f"中間レビューの解析中にエラーが発生しました: {e}")
        return None

def init_session_state():
    if 'generated_titles' not in st.session_state:
        st.session_state.generated_titles = []
    if 'selected_title' not in st.session_state:
        st.session_state.selected_title = None
    if 'selected_title_for_headline' not in st.session_state:
        st.session_state.selected_title_for_headline = None
    if 'headlines' not in st.session_state:
        st.session_state.headlines = None
    if 'title_cache' not in st.session_state:
        st.session_state.title_cache = {}
    if 'seminar_data' not in st.session_state:
        st.session_state.seminar_data = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'extracted_content' not in st.session_state:
        st.session_state.extracted_content = {}
    if 'title_prompt' not in st.session_state:
        st.session_state.title_prompt = TitleGenerator("dummy_key").user_editable_prompt
    if 'headline_prompt' not in st.session_state:
        st.session_state.headline_prompt = HeadlineGenerator("dummy_key").user_editable_prompt
    if 'body_prompt' not in st.session_state:
        st.session_state.body_prompt = BodyGenerator("dummy_key").user_editable_prompt
    if 'generated_body' not in st.session_state:
        st.session_state.generated_body = None
    if 'manual_headlines' not in st.session_state:
        st.session_state.manual_headlines = None
    if 'target_audience' not in st.session_state:
        st.session_state.target_audience = ""
    if 'refined_body_sections' not in st.session_state: 
        st.session_state.refined_body_sections = {}
    if 'product_urls' not in st.session_state:
        st.session_state.product_urls = ["", "", ""]
    if 'file_contents' not in st.session_state:
        st.session_state.file_contents = []
    
    # ペインポイント生成関連
    if 'generated_pain_points' not in st.session_state:
        st.session_state.generated_pain_points = []
    if 'selected_pain_point_index' not in st.session_state:
        st.session_state.selected_pain_point_index = None
    if 'strengths' not in st.session_state:
        st.session_state.strengths = ""
    if 'pain_point_prompt' not in st.session_state:
        st.session_state.pain_point_prompt = PainPointGenerator("dummy_key").user_editable_prompt
    if 'manual_pain_point' not in st.session_state:
        st.session_state.manual_pain_point = None

    # セミナー開催情報用のsession_state（空の初期値）
    if 'seminar_開催日' not in st.session_state:
        st.session_state.seminar_開催日 = ""
    if 'seminar_主催企業' not in st.session_state:
        st.session_state.seminar_主催企業 = ""
    if 'seminar_集客人数' not in st.session_state:
        st.session_state.seminar_集客人数 = ""
    if 'seminar_初稿UP期限' not in st.session_state:
        st.session_state.seminar_初稿UP期限 = ""

    # Slack投稿フォーマット用session_state (共通項目)
    if 'slack_common_参考情報' not in st.session_state:
        st.session_state.slack_common_参考情報 = ""
        
    # 再開モード関連の状態
    if 'resume_mode' not in st.session_state:
        st.session_state.resume_mode = None
    if 'show_pain_point_input' not in st.session_state:
        st.session_state.show_pain_point_input = False
    if 'show_mid_review_input' not in st.session_state:
        st.session_state.show_mid_review_input = False


def main():
    init_session_state()

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        # api_key = os.environ.get("OPENAI_API_KEY") # 環境変数からの読み込み
    except KeyError:
        st.error("OpenAI APIキーが設定されていません")
        return

    st.title("セミナータイトル・告知文 ジェネレーター")
    
    # 再開ボタンセクション
    col1, col2 = st.columns(2)
    with col1:
        if st.button("タイトル生成から再開する"):
            st.session_state.resume_mode = 'from_pain_point'
            st.session_state.show_pain_point_input = True
            st.session_state.show_mid_review_input = False
            st.rerun()
    with col2:
        if st.button("本文生成から再開する"):
            st.session_state.resume_mode = 'from_mid_review'
            st.session_state.show_pain_point_input = False
            st.session_state.show_mid_review_input = True
            st.rerun()
    
    # ペインポイント入力から再開
    if st.session_state.show_pain_point_input:
        st.header("ペインポイントから再開")
        with st.form("pain_point_resume_form"):
            pain_point_text = st.text_area(
                "ペインポイント (最初の行が見出し、残りが詳細になります)", 
                height=150, 
                placeholder="例：\n不確実な市場環境で経営判断を迫られるCXO\n\n市場の変化が速く、競合の動きも読みにくい中で、限られた情報から経営判断を行わなければならないCXOやマネジメント層が増えています。多くの企業がDXや新規事業に取り組む中、間違った判断は大きなリスクとなります。"
            )
            target_audience = st.text_area("ターゲット像", height=100)
            strengths = st.text_area("強み・差別化ポイント", height=100)
            
            submit_button = st.form_submit_button("タイトル生成へ進む")
            
            if submit_button and pain_point_text:
                # ペインポイントを解析
                pain_point = parse_pain_point_input(pain_point_text)
                
                # セッション状態に反映
                st.session_state.target_audience = target_audience
                st.session_state.strengths = strengths
                st.session_state.generated_pain_points = [pain_point]
                st.session_state.selected_pain_point_index = 0
                
                # モードをリセット
                st.session_state.resume_mode = None
                st.session_state.show_pain_point_input = False
                
                # 再読み込みして通常のフローに戻る
                st.rerun()
    
    # 中間レビューから再開
    if st.session_state.show_mid_review_input:
        st.header("中間レビューから再開")
        with st.form("mid_review_resume_form"):
            mid_review_text = st.text_area(
                "中間レビューのフォーマットを貼り付けてください", 
                height=300,
                placeholder="【ペインポイント・タイトル・見出し案の確認依頼】\n\n下記、ご確認をお願いします。\n\n＜対象セミナー＞\n・開催日：...\n..."
            )
            
            # 補足情報入力
            st.subheader("補足情報")
            col1, col2 = st.columns(2)
            with col1:
                開催日 = st.date_input("開催日")
                st.session_state.seminar_開催日 = 開催日.strftime('%-m/%-d')
                集客人数 = st.text_input("集客人数")
                st.session_state.seminar_集客人数 = 集客人数
            with col2:
                主催企業 = st.text_input("主催企業")
                st.session_state.seminar_主催企業 = 主催企業
                初稿UP期限 = st.date_input("初稿UP期限")
                st.session_state.seminar_初稿UP期限 = 初稿UP期限.strftime('%-m/%-d(%a)')
            
            submit_button = st.form_submit_button("本文生成へ進む")
            
            if submit_button and mid_review_text:
                # 中間レビューを解析
                result = parse_mid_review_format(mid_review_text)
                if result:
                    # 必要な状態を設定
                    st.session_state.target_audience = result.get("target", "")
                    st.session_state.strengths = result.get("strengths", "")
                    
                    # ペインポイント設定
                    pain_point = PainPoint(
                        headline=result.get("pain_point_headline", ""),
                        description=result.get("pain_point_description", "")
                    )
                    st.session_state.generated_pain_points = [pain_point]
                    st.session_state.selected_pain_point_index = 0
                    
                    # タイトル設定
                    st.session_state.selected_title = result.get("seminar_title", "")
                    st.session_state.selected_title_for_headline = result.get("seminar_title", "")
                    
                    # 見出し設定
                    headlines = HeadlineSet(
                        background=result.get("headline_background", ""),
                        problem=result.get("headline_problem", ""),
                        solution=result.get("headline_solution", "")
                    )
                    st.session_state.headlines = headlines
                    st.session_state.manual_headlines = headlines
                    
                    # モードをリセット
                    st.session_state.resume_mode = None
                    st.session_state.show_mid_review_input = False
                    
                    # 再読み込みして本文生成から開始
                    st.rerun()
                else:
                    st.error("中間レビューの解析に失敗しました。正しいフォーマットで入力してください。")

    if st.session_state.seminar_data is None:
        with st.spinner("セミナーデータを読み込んでいます..."):
            df = load_seminar_data()
            if df is not None:
                st.session_state.seminar_data = df
                st.session_state.evaluator = SeminarTitleEvaluator(df)
                st.success("データを正常に読み込みました！")
            else:
                st.error("データの読み込みに失敗しました。")
                return

    model_name = "gpt-4o"
    pain_point_generator = PainPointGenerator(api_key, model=model_name)
    title_generator = TitleGenerator(api_key, model=model_name)
    headline_generator = HeadlineGenerator(api_key, model=model_name)
    body_generator = BodyGenerator(api_key, model=model_name)
    cache = InMemoryCache()

    # 中間レビューから再開する場合、本文生成ステップに直接ジャンプ
    if st.session_state.resume_mode == 'from_mid_review' and st.session_state.manual_headlines:
        st.header("Step 5: 本文生成")
        # 本文生成の処理...
    # 通常のフローに戻る
    else:
        st.header("Step 1: 基本情報入力")

        # 基本情報入力 - メインカラムを2つに分割
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 製品URL入力 - 最大3つ
            st.subheader("製品URL (最大3つ)")
            url_container = st.container()
            
            for i in range(3):
                product_url = url_container.text_input(f"製品URL {i+1}", key=f"product_url_{i}", value=st.session_state.product_urls[i])
                st.session_state.product_urls[i] = product_url
                
                if product_url:
                    with st.spinner(f"URL {i+1} からコンテンツを取得中..."):
                        extractor = URLContentExtractor()
                        content = extractor.extract_with_trafilatura(product_url)
                        if content and not content.error:
                            st.success(f"製品情報 {i+1} を取得しました")
                            with st.expander(f"取得した製品情報 {i+1}"):
                                st.write("**タイトル:**", content.title)
                                st.write("**説明:**", content.description)
                                st.write("**詳細:**", content.main_content[:300] + "...")
                        elif product_url:  # URLが入力されているが、取得に失敗した場合
                            st.error(f"コンテンツの取得に失敗しました: {content.error if content else '不明なエラー'}")
        
        with col2:
            # 右側のカラム - 強み・差別化ポイント、ターゲット像
            st.session_state.strengths = st.text_area("強み・差別化ポイント", value=st.session_state.strengths, height=100)
            target_audience = st.text_area("ターゲット像", height=100)
            st.session_state.target_audience = target_audience

        # ファイルアップロード部分 - 最大5ファイル
        st.subheader("ファイル (最大5つまでアップロード可能)")
        uploaded_files = st.file_uploader("ファイルをアップロード", type=['txt', 'pdf', 'docx'], accept_multiple_files=True)
        
        # ファイル内容を保存する配列をリセット
        file_contents = []
        
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files[:5]):  # 最大5ファイルまで処理
                try:
                    file_content = ""
                    if uploaded_file.type == "text/plain":
                        file_content = uploaded_file.getvalue().decode('utf-8')
                    elif uploaded_file.type == "application/pdf":
                        reader = PdfReader(uploaded_file)
                        file_content = ""
                        for page in reader.pages:
                            file_content += page.extract_text()
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        document = Document(uploaded_file)
                        file_content = "\n".join([para.text for para in document.paragraphs])
                    else:
                        st.error(f"未対応のファイルタイプです: {uploaded_file.type}")
                        continue
                    
                    file_contents.append(file_content)
                    st.success(f"ファイル {i+1}: {uploaded_file.name} を正常に読み込みました")
                    
                    with st.expander(f"ファイル {i+1}: {uploaded_file.name} の内容"):
                        st.text(file_content[:1000] + ("..." if len(file_content) > 1000 else ""))
                        
                except Exception as e:
                    st.error(f"ファイル {uploaded_file.name} の読み込みでエラーが発生しました: {e}")
        
        # ファイル内容をセッションに保存
        st.session_state.file_contents = file_contents

    # Step 2: ペインポイント生成
    if not st.session_state.resume_mode == 'from_mid_review':
        st.header("Step 2: ペインポイント生成")

        with st.expander("ペインポイント生成プロンプトの編集", expanded=False):
            st.session_state.pain_point_prompt = st.text_area(
                "プロンプトテンプレート",
                value=st.session_state.pain_point_prompt,
                height=400
            )

        if st.button("ペインポイントを生成", key="generate_pain_points"):
            if not st.session_state.target_audience:
                st.error("ターゲット像を入力してください")
            elif not st.session_state.strengths:
                st.error("強み・差別化ポイントを入力してください")
            else:
                with st.spinner("ペインポイントを生成中..."):
                    # 空でないURLのみを渡す
                    valid_urls = [url for url in st.session_state.product_urls if url]
                    
                    pain_points = pain_point_generator.generate_pain_points(
                        st.session_state.target_audience,
                        st.session_state.strengths,
                        st.session_state.pain_point_prompt,
                        valid_urls,
                        st.session_state.file_contents
                    )

                    if pain_points:
                        st.session_state.generated_pain_points = pain_points
                        st.session_state.selected_pain_point_index = 0  # 最初のペインポイントを選択
                        st.rerun()

        # 生成されたペインポイントの表示（更新版 - 選択をより明確に、見出しと詳細を統合）
        if st.session_state.generated_pain_points:
            st.subheader("生成されたペインポイント")
            
            for i, pain_point in enumerate(st.session_state.generated_pain_points):
                is_selected = i == st.session_state.selected_pain_point_index
                
                # 選択状態に応じたスタイル適用
                container_style = "selected-pain-point" if is_selected else "unselected-pain-point"
                
                with st.container():
                    st.markdown(f'<div class="{container_style}">', unsafe_allow_html=True)
                    cols = st.columns([1, 5, 1])
                    
                    with cols[0]:
                        # 選択ボタン - 状態に応じてラベルとスタイルを変更
                        if st.button("✓ 選択中" if is_selected else "選択", 
                                    key=f"pain_point_select_{i}",
                                    type="primary" if is_selected else "secondary"):
                            st.session_state.selected_pain_point_index = i
                            st.rerun()
                    
                    with cols[1]:
                        # 見出しと詳細を統合表示
                        st.markdown(f"**{pain_point.headline}**\n\n{pain_point.description}")
                    
                    with cols[2]:
                        修正プロンプト = st.text_area("修正依頼", key=f"refine_pain_point_{i}", height=70, label_visibility="collapsed", placeholder="例：もっと具体的に")
                        if st.button("修正", key=f"refine_pain_point_button_{i}"):
                            with st.spinner("ペインポイント修正中..."):
                                combined_text = pain_point.combined_text()
                                refined_pain_point = pain_point_generator.refine_pain_point(combined_text, 修正プロンプト)
                                if refined_pain_point:
                                    refined_headline = refined_pain_point.headline
                                    refined_description = refined_pain_point.description
                                    
                                    st.session_state.generated_pain_points[i] = PainPoint(
                                        headline=refined_headline,
                                        description=refined_description
                                    )
                                    st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            # 手動ペインポイント追加（統合版）
            st.subheader("手動ペインポイント追加")
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # 入力欄を統合
                manual_pain_point_text = st.text_area(
                    "ペインポイント (最初の行が見出し、残りが詳細になります)", 
                    key="manual_pain_point_text", 
                    height=150,
                    placeholder="例：\n不確実な市場環境で経営判断を迫られるCXO\n\n市場の変化が速く、競合の動きも読みにくい中で、限られた情報から経営判断を行わなければならないCXOやマネジメント層が増えています。多くの企業がDXや新規事業に取り組む中、間違った判断は大きなリスクとなります。"
                )
            
            with col2:
                if st.button("追加", key="add_manual_pain_point") and manual_pain_point_text:
                    # 入力テキストを解析してペインポイントを作成
                    new_pain_point = parse_pain_point_input(manual_pain_point_text)
                    st.session_state.generated_pain_points.append(new_pain_point)
                    st.session_state.selected_pain_point_index = len(st.session_state.generated_pain_points) - 1
                    st.rerun()
            
    # Step 3: タイトル生成
    if st.session_state.generated_pain_points and st.session_state.selected_pain_point_index is not None and not st.session_state.resume_mode == 'from_mid_review':
        st.header("Step 3: タイトル生成")
        
        with st.expander("タイトル生成プロンプトの編集", expanded=False):
            st.session_state.title_prompt = st.text_area(
                "プロンプトテンプレート",
                value=st.session_state.title_prompt,
                height=400
            )

        selected_pain_point = st.session_state.generated_pain_points[st.session_state.selected_pain_point_index]
        
        if st.button("タイトルを生成", key="generate_titles"):
            context = f"""
強み・差別化ポイント: {st.session_state.strengths}
"""
            with st.spinner("タイトルを生成中..."):
                # 空でないURLのみを渡す
                valid_urls = [url for url in st.session_state.product_urls if url]
                
                titles = title_generator.generate_titles(
                    context,
                    st.session_state.target_audience,
                    selected_pain_point,
                    st.session_state.title_prompt,
                    valid_urls,
                    st.session_state.file_contents
                )
                
                st.session_state.generated_titles = []
                for title in titles:
                    main_title = title.get("main_title", "")
                    sub_title = title.get("sub_title", "")
                    full_title = f"{main_title} - {sub_title}"
                    cached_eval = cache.get_evaluation(full_title)
                    if cached_eval:
                        evaluation = cached_eval
                    else:
                        analysis = st.session_state.evaluator.evaluate_title(full_title)
                        evaluation = TitleEvaluation(
                            speed=analysis.predicted_speed,
                            grade=analysis.grade,
                            comment=analysis.evaluation_comment
                        )
                        cache.set_evaluation(full_title, evaluation)
                    st.session_state.generated_titles.append(
                        GeneratedTitle(
                            main_title=main_title,
                            sub_title=sub_title,
                            evaluation=evaluation,
                            original_main_title=main_title,
                            original_sub_title=sub_title
                        )
                    )

        # タイトル評価・選択
        if st.session_state.generated_titles:
            st.subheader("生成タイトル")
            for i, gen_title in enumerate(st.session_state.generated_titles):
                cols = st.columns([0.5, 2, 2, 1, 1, 2, 1])
                with cols[0]:
                    if st.radio(
                        "選択",
                        ["✓"],
                        key=f"radio_{i}",
                        label_visibility="collapsed"
                    ):
                        st.session_state.selected_title = f"{gen_title.main_title} - {gen_title.sub_title}"
                with cols[1]:
                    st.write(f"**メインタイトル:** {gen_title.main_title}")
                with cols[2]:
                    st.write(f"**サブタイトル:** {gen_title.sub_title}")
                with cols[3]:
                    st.metric("集客速度", f"{gen_title.evaluation.speed:.1f}")
                with cols[4]:
                    grade_colors = {"A": "green", "B": "orange", "C": "red"}
                    grade_color = grade_colors.get(gen_title.evaluation.grade, "gray")
                    st.markdown(
                        f'<p style="color: {grade_color}; font-weight: bold; text-align: center;">評価: {gen_title.evaluation.grade}</p>',
                        unsafe_allow_html=True
                    )
                with cols[5]:
                    st.write(f"**評価:** {gen_title.evaluation.comment}")
                with cols[6]:
                    修正プロンプト = st.text_area("修正依頼", key=f"refine_prompt_{i}", height=70, label_visibility="collapsed", placeholder="例：もっと具体的に")
                    if st.button("修正", key=f"refine_button_{i}"):
                        with st.spinner("タイトル修正中..."):
                            refined_title = title_generator.refine_title(gen_title.main_title, gen_title.sub_title, 修正プロンプト)
                            if refined_title:
                                refined_main = refined_title.main_title
                                refined_sub = refined_title.sub_title

                                full_refined_title = f"{refined_main} - {refined_sub}"
                                analysis = st.session_state.evaluator.evaluate_title(full_refined_title)
                                evaluation = TitleEvaluation(
                                    speed=analysis.predicted_speed,
                                    grade=analysis.grade,
                                    comment=analysis.evaluation_comment
                                )
                                st.session_state.generated_titles[i] = GeneratedTitle(
                                    main_title=refined_main,
                                    sub_title=refined_sub,
                                    evaluation=evaluation,
                                    original_main_title=gen_title.original_main_title,
                                    original_sub_title=gen_title.original_sub_title
                                )
                                st.rerun()

            # 手動タイトル評価
            st.subheader("手動タイトル評価")
            col1, col2 = st.columns([4, 1])
            with col1:
                manual_main_title = st.text_input("メインタイトル", key="manual_main_title")
                manual_sub_title = st.text_input("サブタイトル", key="manual_sub_title")
            with col2:
                if st.button("評価する", key="evaluate_manual") and manual_main_title:
                    with st.spinner("評価中..."):
                        try:
                            full_title = f"{manual_main_title} - {manual_sub_title}"
                            cached_eval = cache.get_evaluation(full_title)
                            if cached_eval:
                                evaluation = cached_eval
                            else:
                                analysis = st.session_state.evaluator.evaluate_title(full_title)
                                evaluation = TitleEvaluation(
                                    speed=analysis.predicted_speed,
                                    grade=analysis.grade,
                                    comment=analysis.evaluation_comment
                                )
                                cache.set_evaluation(full_title, evaluation)
                            st.session_state.generated_titles.append(
                                GeneratedTitle(
                                    main_title=manual_main_title,
                                    sub_title=manual_sub_title,
                                    evaluation=evaluation,
                                    original_main_title=manual_main_title,
                                    original_sub_title=manual_sub_title
                                )
                            )

                            display_evaluation_details(full_title, st.session_state.evaluator)
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")

            # Step 4: 見出し生成
            if st.session_state.selected_title:
                st.header("Step 4: 見出し生成")

                available_titles = []
                for gen_title in st.session_state.generated_titles:
                    full_title = f"{gen_title.main_title} - {gen_title.sub_title}"
                    available_titles.append(full_title)

                st.session_state.selected_title_for_headline = st.selectbox(
                    "見出しを生成するタイトルを選択してください",
                    options=available_titles,
                    index=available_titles.index(st.session_state.selected_title) if st.session_state.selected_title in available_titles else 0
                )

                with st.expander("見出し生成プロンプトの編集", expanded=False):
                    st.session_state.headline_prompt = st.text_area(
                        "プロンプトテンプレート",
                        value=st.session_state.headline_prompt,
                        height=400
                    )

                # 選択されたペインポイントを取得
                selected_pain_point = st.session_state.generated_pain_points[st.session_state.selected_pain_point_index]

                if st.button("見出しを生成", key="generate_headlines"):
                    with st.spinner("見出しを生成中..."):
                        try:
                            headlines = headline_generator.generate_headlines(
                                st.session_state.selected_title_for_headline,
                                st.session_state.target_audience,
                                selected_pain_point,
                                st.session_state.headline_prompt
                            )
                            st.session_state.headlines = headlines
                            st.session_state.manual_headlines = headlines
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")

                if st.session_state.manual_headlines:
                    st.subheader("生成された見出し（編集可能）")

                    background = st.text_area(
                        "背景",
                        value=st.session_state.manual_headlines.background,
                        key="edit_background"
                    )
                    problem = st.text_area(
                        "課題",
                        value=st.session_state.manual_headlines.problem,
                        key="edit_problem"
                    )
                    solution = st.text_area(
                        "解決策",
                        value=st.session_state.manual_headlines.solution,
                        key="edit_solution"
                    )

                    st.session_state.manual_headlines = HeadlineSet(
                        background=background,
                        problem=problem,
                        solution=solution
                    )

                    # [移動] 中間レビュー用フォーマット出力をここに移動
                    st.subheader("中間レビュー: ペインポイント・タイトル・見出し確認依頼")
                    
                    with st.expander("中間レビュー確認依頼フォーマット入力", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            # 開催日を日付選択に変更
                            開催日 = st.date_input("開催日", key="mid_review_開催日")
                            st.session_state.seminar_開催日 = 開催日.strftime('%-m/%-d')  # 月/日 形式で保存
                            st.session_state.seminar_集客人数 = st.text_input("集客人数", key="mid_review_集客人数")
                        with col2:
                            st.session_state.seminar_主催企業 = st.text_input("主催企業", key="mid_review_主催企業")
                            # 初稿UP期限を日付選択に変更
                            初稿UP期限 = st.date_input("初稿UP期限", key="mid_review_初稿UP期限")
                            st.session_state.seminar_初稿UP期限 = 初稿UP期限.strftime('%-m/%-d(%a)')  # 月/日(曜日) 形式で保存

                    # 選択されたタイトルとペインポイントを取得
                    selected_pain_point = st.session_state.generated_pain_points[st.session_state.selected_pain_point_index]
                    product_urls_for_slack = "\n".join([url for url in st.session_state.product_urls if url])
                    
                    if st.button("中間レビュー Slack生成", key="generate_mid_review_format"):
                        mid_review_format = generate_mid_review_format(
                            st.session_state.seminar_開催日,
                            st.session_state.seminar_主催企業,
                            st.session_state.seminar_集客人数,
                            st.session_state.seminar_初稿UP期限,
                            product_urls_for_slack,
                            st.session_state.target_audience,
                            st.session_state.strengths,
                            selected_pain_point,
                            st.session_state.selected_title_for_headline,
                            st.session_state.manual_headlines
                        )
                        st.subheader("生成された中間レビュー (Slackへコピペできます)")
                        st.code(mid_review_format, language="text")

    # Step 5: 本文生成 - 中間レビューから再開する場合もこの部分を表示
    if ((st.session_state.manual_headlines and st.session_state.selected_title_for_headline) or 
        (st.session_state.resume_mode == 'from_mid_review' and st.session_state.manual_headlines)):
        
        st.header("Step 5: 本文生成")

        with st.expander("本文生成プロンプトの編集", expanded=False):
            st.session_state.body_prompt = st.text_area(
                "本文生成プロンプトテンプレート",
                value=st.session_state.body_prompt,
                height=400
            )

        selected_pain_point = st.session_state.generated_pain_points[st.session_state.selected_pain_point_index]

        if st.button("本文を生成", key="generate_body"):
            with st.spinner("本文を生成中..."):
                try:
                    generated_body = body_generator.generate_body(
                        st.session_state.selected_title_for_headline,
                        st.session_state.manual_headlines,
                        st.session_state.target_audience,
                        selected_pain_point,
                        st.session_state.body_prompt
                    )
                    st.session_state.generated_body = generated_body
                    
                    # 生成された本文をセクションごとに分割 - 改良版パーサーを使用
                    sections = body_generator.parse_body_sections(generated_body, st.session_state.manual_headlines)
                    if sections:
                        st.session_state.refined_body_sections = sections
                    else:
                        st.error("本文セクションの分割に失敗しました。代替方法を試みます。")
                        # 代替方法: 単純にセクション分割
                        try:
                            sections = re.split(r'##\s+', generated_body)
                            if len(sections) >= 4:  # 最初の分割は空になるはず
                                sections = sections[1:]  # 最初の空セクションを削除
                                st.session_state.refined_body_sections = {
                                    "background": sections[0].strip(),
                                    "problem": sections[1].strip() if len(sections) > 1 else "",
                                    "solution": sections[2].strip() if len(sections) > 2 else ""
                                }
                            else:
                                st.error("本文の分割に失敗しました。手動での編集が必要です。")
                                st.session_state.refined_body_sections = {
                                    "background": "",
                                    "problem": "",
                                    "solution": ""
                                }
                        except Exception as parse_error:
                            st.error(f"代替パース方法でもエラーが発生しました: {parse_error}")
                            st.session_state.refined_body_sections = {}

                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

        if st.session_state.generated_body:
            st.subheader("生成された本文")

            if st.session_state.refined_body_sections:
                # 背景セクション
                st.markdown(f"## {st.session_state.manual_headlines.background}")
                st.markdown(st.session_state.refined_body_sections.get("background", ""))
                col1, col2 = st.columns([4, 1])
                with col1:
                    background_prompt = st.text_area("修正指示 (背景)", key="refine_body_prompt_background", height=70, placeholder="例：もっと具体的に")
                with col2:
                    if st.button("修正", key="refine_body_button_background"):
                        with st.spinner("背景セクション修正中..."):
                            refined_section = body_generator.refine_body_section(
                                st.session_state.refined_body_sections["background"], 
                                background_prompt, 
                                "背景"
                            )
                            if refined_section:
                                st.session_state.refined_body_sections["background"] = refined_section.refined_text
                                st.rerun()

                # 課題セクション
                st.markdown(f"## {st.session_state.manual_headlines.problem}")
                st.markdown(st.session_state.refined_body_sections.get("problem", ""))
                col1, col2 = st.columns([4, 1])
                with col1:
                    problem_prompt = st.text_area("修正指示 (課題)", key="refine_body_prompt_problem", height=70, placeholder="例：もっと具体的に")
                with col2:
                    if st.button("修正", key="refine_body_button_problem"):
                        with st.spinner("課題セクション修正中..."):
                            refined_section = body_generator.refine_body_section(
                                st.session_state.refined_body_sections["problem"], 
                                problem_prompt, 
                                "課題"
                            )
                            if refined_section:
                                st.session_state.refined_body_sections["problem"] = refined_section.refined_text
                                st.rerun()

                # 解決策セクション
                st.markdown(f"## {st.session_state.manual_headlines.solution}")
                st.markdown(st.session_state.refined_body_sections.get("solution", ""))
                col1, col2 = st.columns([4, 1])
                with col1:
                    solution_prompt = st.text_area("修正指示 (解決策)", key="refine_body_prompt_solution", height=70, placeholder="例：もっと具体的に")
                with col2:
                    if st.button("修正", key="refine_body_button_solution"):
                        with st.spinner("解決策セクション修正中..."):
                            refined_section = body_generator.refine_body_section(
                                st.session_state.refined_body_sections["solution"], 
                                solution_prompt, 
                                "解決策"
                            )
                            if refined_section:
                                st.session_state.refined_body_sections["solution"] = refined_section.refined_text
                                st.rerun()

            else: # refined_body_sections がない場合（エラー発生時など）はプレーンテキストで全体を表示
                st.write(st.session_state.generated_body)


            # Step 6: Slack投稿フォーマット生成
            st.header("Step 6: Slack投稿フォーマット生成")

            slack_format_tab = st.tabs(["企画案レビュー"])

            with slack_format_tab[0]:
                st.subheader("企画案レビュー Slack投稿フォーマット")
                with st.expander("企画案レビュー Slack投稿フォーマット入力", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        # 開催日を日付選択に変更
                        開催日 = st.date_input("開催日", key="plan_開催日")
                        st.session_state.seminar_開催日 = 開催日.strftime('%-m/%-d')  # 月/日 形式で保存
                        st.session_state.seminar_集客人数 = st.text_input("集客人数", key="plan_集客人数")
                    with col2:
                        st.session_state.seminar_主催企業 = st.text_input("主催企業", key="plan_主催企業")
                        # 初稿UP期限を日付選択に変更
                        初稿UP期限 = st.date_input("初稿UP期限", key="plan_初稿UP期限")
                        st.session_state.seminar_初稿UP期限 = 初稿UP期限.strftime('%-m/%-d(%a)')  # 月/日(曜日) 形式で保存

                # 複数のURLに対応するための対応
                product_urls_for_slack = "\n".join([url for url in st.session_state.product_urls if url])
                selected_pain_point = st.session_state.generated_pain_points[st.session_state.selected_pain_point_index]

                if st.button("企画案レビュー Slack投稿フォーマット生成", key="generate_slack_plan_format"):
                    plan_format_text = generate_plan_review_format(
                        st.session_state.seminar_開催日,
                        st.session_state.seminar_主催企業,
                        st.session_state.seminar_集客人数,
                        st.session_state.seminar_初稿UP期限,
                        product_urls_for_slack,  # 複数URLをまとめたもの
                        st.session_state.selected_title_for_headline,
                        st.session_state.manual_headlines.background,
                        st.session_state.manual_headlines.problem,
                        st.session_state.manual_headlines.solution,
                        st.session_state.target_audience,
                        selected_pain_point,
                        st.session_state.strengths,
                        st.session_state.refined_body_sections.get("background", ""),
                        st.session_state.refined_body_sections.get("problem", ""),
                        st.session_state.refined_body_sections.get("solution", "")
                    )
                    st.subheader("生成された企画案レビュー Slack投稿フォーマット (Slackへコピペできます)")
                    st.code(plan_format_text, language="text")

if __name__ == "__main__":
    main()
