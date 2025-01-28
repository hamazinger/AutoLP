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
    page_title="セミナタイトルジェネレーター",
    layout="wide"
)

# 右下の開発者プロフィールリンクやフッター非表示用CSS
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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
class TitleAnalysis:
    predicted_speed: float
    grade: str
    attractive_words: List[str]
    has_specific_problem: bool
    has_exclamation: bool
    title_length: int
    category_score: float
    reasoning: Dict[str, str]
    evaluation_comment: str  # 追加：評価コメント

@dataclass
class TitleEvaluation:
    speed: float
    grade: str
    comment: str  # 追加：評価コメント
    timestamp: str = datetime.now().isoformat()

@dataclass
class GeneratedTitle:
    main_title: str
    sub_title: str
    evaluation: TitleEvaluation
    original_main_title: str  # 修正前のメインタイトルを保持
    original_sub_title: str    # 修正前のサブタイトルを保持

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
        # except Exception as e:
        #     return self.extract_with_fallback(url, str(e))

    # def extract_with_fallback(self, url: str, prev_error: str) -> WebContent:
    #     try:
    #         # フォールバック処理: requestsで直接HTML取得
    #         response = requests.get(url, headers=self.headers, timeout=10)
    #         response.raise_for_status()

    #         # 自動エンコード判定
    #         response.encoding = response.apparent_encoding

    #         soup = BeautifulSoup(response.text, 'html.parser')
    #         title = soup.title.string if soup.title else ""
    #         meta_desc = soup.find('meta', {'name': 'description'})
    #         description = meta_desc['content'] if meta_desc else ""
    #         body_content = '\n'.join([p.get_text() for p in soup.find_all('p')])

    #         return WebContent(
    #             title=title,
    #             description=description,
    #             main_content=body_content
    #         )
    #     except Exception as e:
    #         # フォールバックも失敗
    #         return WebContent(
    #             title="",
    #             description="",
    #             main_content="",
    #             error=f"フォールバックも失敗しました: {str(e)} (前エラー: {prev_error})"
    #         )

# Pydanticモデルの定義
class RefinedTitles(BaseModel):
    main_title: str = Field(description="修正後のメインタイトル")
    sub_title: str = Field(description="修正後のサブタイトル")

    # 非推奨の.dict()メソッドを削除し、.model_dump()を使用
    def model_dump(self) -> Dict[str, str]:
        return {
            "main_title": self.main_title,
            "sub_title": self.sub_title,
        }

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

# ターゲット像
{target}

# Steps

1. 入力情報とターゲット像から製品の特徴とターゲットのペインポイントを理解する
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

    def generate_titles(self, context: str, target: str, prompt_template: str = None, product_url: str = None, file_content: str = None) -> List[Dict[str, str]]:
        additional_context = ""
        if product_url:
            content = self.url_extractor.extract_with_trafilatura(product_url)
            if content and not content.error:
                additional_context += f"""
製品タイトル: {content.title}
製品説明: {content.description}
製品詳細: {content.main_content[:1000]}
"""
            else:
                st.warning(f"製品情報の取得に失敗しました: {content.error if content else '不明なエラー'}")

        if file_content:
            additional_context += f"""
アップロードされたファイルの内容:
{file_content}
"""

        prompt = f"""
# 入力情報
{context}
""" + (prompt_template or self.user_editable_prompt).format(target=target) + additional_context + self.fixed_output_instructions

        result_text = None  # result_text を None で初期化

        try:
            # 新しい書き方で API を呼び出す
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

        except Exception as e:  # すべての例外をキャッチ
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

class SeminarTitleEvaluator:
    def __init__(self, seminar_data: pd.DataFrame):
        self.df = seminar_data
        self._initialize_analytics()

    def _initialize_analytics(self):
        high_performing = self.df[self.df['Acquisition_Speed'] >= 2.5]
        self.attractive_words = self._extract_effective_keywords(high_performing)
        self.category_speeds = self.df.groupby('Major_Category')['Acquisition_Speed'].mean()
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
        """評価コメントを生成する新しいメソッド"""
        comments = []

        # スコアに基づくコメント
        if analysis_data["predicted_speed"] >= 2.5:
            comments.append("高い集客が期待できます")
        elif analysis_data["predicted_speed"] >= 1.8:
            comments.append("一定の集客が見込めます")
        else:
            comments.append("改善の余地があります")

        # キーワードに基づくコメント
        if analysis_data["attractive_words"]:
            comments.append("効果的なキーワードが含まれています")
        else:
            comments.append("効果的なキーワードの追加を検討してください")

        # 長さに基づくコメント
        if analysis_data["title_length"] > 40:
            comments.append("タイトルを短くすることを推奨します")

        # 問題提起の有無
        if not analysis_data["has_specific_problem"]:
            comments.append("具体的な課題や問題提起の追加を検討してください")

        return "。".join(comments)

    def evaluate_title(self, title: str, category: str = None) -> TitleAnalysis:
        base_score = self._calculate_base_score(title)

        category_score = 0.0
        if category and category in self.category_speeds:
            category_avg = self.category_speeds[category]
            category_score = 0.3 if category_avg > 2.5 else (
                0.2 if category_avg > 2.0 else 0.1
            )

        final_score = min(max(base_score + category_score, 1.0), 3.0)

        matching_words = [word for word in self.attractive_words if word in title]
        has_problem = any(indicator in title for indicator in self.problem_indicators)

        reasoning = {
            "keywords": f"効果的なキーワード: {', '.join(matching_words) if matching_words else '該当なし'}",
            "title_length": f"タイトルの長さ: {len(title)}文字 （{'適切' if len(title) <= 40 else '長い'}）",
            "problem_indication": f"問題提起: {'あり' if has_problem else 'なし'}",
            "exclamation": f"感嘆符: {'あり（減点）' if '!' in title or '！' in title else 'なし'}",
            "category": f"カテゴリ評価: {category if category else '未指定'} (スコア: {category_score:.1f})",
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
            category_score=category_score,
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

class HeadlineGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fixed_prompt_part = """
「『{title}』というタイトルのイベントを企画しており、その告知文を作成します。 告知文を作成する前に、以下の内容でその見出しを３つ作成してください。それぞれの見出しは簡潔な文章としてください。 」
"""
        self.user_editable_prompt = """
見出し1：このセミナーを開催する、社会や企業の背景
見出し2：このセミナーで訴求したい、課題、問題、悩み、不安
見出し3：上記課題の解決の方向性
- **ターゲット像を意識する**

# ターゲット像
{target}
"""
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください：
{
    "background": "背景の見出し",
    "problem": "課題の見出し",
    "solution": "解決策の見出し"
}
"""

    def generate_headlines(self, title: str, target: str, prompt_template: str = None) -> HeadlineSet:
        """タイトルに基づいて見出しを生成"""
        prompt = self.fixed_prompt_part.format(title=title) + (prompt_template or self.user_editable_prompt).format(target=target) + self.fixed_output_instructions

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
        self.fixed_prompt_part = """
以下のセミナータイトルと見出しに基づいて、本文を生成してください：
- 各見出しは本文中に明示してください。明確に見出しであることがわかるマークダウンの書式（見出しレベル4）を用いてください。

タイトル：「{title}」
{background}
{problem}」
{solution}
"""
        self.user_editable_prompt = """
以下の制約条件と入力情報を踏まえて本文を生成してください。

# 制約条件
- 各見出しセクションは最低300文字以上とし、3文以内でまとめてください（句読点で区切られた3文以内）。
- 全文で1000文字以内に収めてください。
- 本文中では箇条書きを使用しないでください。
- 3つの見出しを通して、一連のストーリーとして流れを持たせてください。
- セミナー内容の紹介および参加を促す表現は、3つ目の見出しのセクションでのみ行ってください。
- 重要なキーワードは本文中に必ず含めてください。
- あくまでセミナー集客用の文章であることを念頭に、魅力的かつ説得力のある内容にしてください。
- **ターゲット像を意識する**

# ターゲット像
{target}
"""

    def generate_body(self, title: str, headlines: HeadlineSet, target: str, prompt_template: str = None) -> str:
        prompt = self.fixed_prompt_part.format(
            title=title,
            background=headlines.background,
            problem=headlines.problem,
            solution=headlines.solution
        ) + (prompt_template or self.user_editable_prompt).format(target=target)

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
        if 'Major_Category' not in df.columns:
            st.error("Major_Categoryカラムが見つかりません")
            st.write("利用可能なカラム:", df.columns.tolist())
            return None

        return df
    except Exception as e:
        st.error(f"データの読み込みでエラーが発生しました: {str(e)}")
        return None

def display_evaluation_details(title: str, evaluator: SeminarTitleEvaluator):
    analysis = evaluator.evaluate_title(
        title,
        st.session_state.selected_category
    )

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

def init_session_state():
    if 'generated_titles' not in st.session_state:
        st.session_state.generated_titles = []
    if 'selected_title' not in st.session_state:
        st.session_state.selected_title = None
    if 'selected_title_for_headline' not in st.session_state:
        st.session_state.selected_title_for_headline = None
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'headlines' not in st.session_state:
        st.session_state.headlines = None
    if 'title_cache' not in st.session_state:
        st.session_state.title_cache = {}
    if 'seminar_data' not in st.session_state:
        st.session_state.seminar_data = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'available_categories' not in st.session_state:
        st.session_state.available_categories = []
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
    if 'target_audience' not in st.session_state: # ターゲット像をsession_stateに追加
        st.session_state.target_audience = ""

def main():
    init_session_state()

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OpenAI APIキーが設定されていません")
        return

    st.title("セミナータイトルジェネレーター")

    if st.session_state.seminar_data is None:
        with st.spinner("セミナーデータを読み込んでいます..."):
            df = load_seminar_data()
            if df is not None:
                try:
                    categories = df['Major_Category'].dropna().unique().tolist()
                    st.session_state.available_categories = sorted(categories)
                    st.session_state.seminar_data = df
                    st.session_state.evaluator = SeminarTitleEvaluator(df)
                    st.success("データを正常に読み込みました！")
                except Exception as e:
                    st.error(f"カテゴリデータの処理中にエラーが発生しました: {e}")
                    return
            else:
                st.error("データの読み込みに失敗しました。")
                return

    model_name = "gpt-4o"
    title_generator = TitleGenerator(api_key, model=model_name)
    headline_generator = HeadlineGenerator(api_key, model=model_name)
    body_generator = BodyGenerator(api_key, model=model_name)
    cache = InMemoryCache()

    st.header("Step 1: 基本情報入力")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1]) # col4 を追加
    with col1:
        product_url = st.text_input("製品URL")
        if product_url:
            with st.spinner("URLからコンテンツを取得中..."):
                extractor = URLContentExtractor()
                content = extractor.extract_with_trafilatura(product_url)
                if content and not content.error:
                    st.success("製品情報を取得しました")
                    with st.expander("取得した製品情報"):
                        st.write("**タイトル:**", content.title)
                        st.write("**説明:**", content.description)
                        st.write("**詳細:**", content.main_content[:500] + "...")
                else:
                    st.error(f"コンテンツの取得に失敗しました: {content.error if content else '不明なエラー'}")
    with col2:
        pain_points = st.text_area("ペインポイント")
    with col3:
        category = st.selectbox(
            "カテゴリ",
            options=st.session_state.available_categories
        )
        st.session_state.selected_category = category
    with col4: # col4 にターゲット像入力欄を追加
        target_audience = st.text_area("ターゲット像", height=80)
        st.session_state.target_audience = target_audience

    uploaded_file = st.file_uploader("ファイルをアップロード", type=['txt', 'pdf', 'docx'])
    file_content = ""
    if uploaded_file is not None:
        try:
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
            st.success("ファイルを正常に読み込みました")
            with st.expander("アップロードされたファイルの内容"):
                st.write(file_content)
        except Exception as e:
            st.error(f"ファイルの読み込みでエラーが発生しました: {e}")

    with st.expander("タイトル生成プロンプトの編集", expanded=False):
        st.session_state.title_prompt = st.text_area(
            "プロンプトテンプレート",
            value=st.session_state.title_prompt,            height=400
        )

    if st.button("タイトルを生成", key="generate_titles"):
        context = f"""
ペインポイント: {pain_points}
カテゴリ: {category}
"""
        with st.spinner("タイトルを生成中..."):
            try:
                titles = title_generator.generate_titles(
                    context,
                    st.session_state.target_audience, # ターゲット像を引数に追加
                    st.session_state.title_prompt,
                    product_url,
                    file_content
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
                        analysis = st.session_state.evaluator.evaluate_title(full_title, category)
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
                            original_main_title=main_title,  # 元のタイトルを保存
                            original_sub_title=sub_title    # 元のタイトルを保存
                        )
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

    if st.session_state.generated_titles:
        st.header("Step 2: タイトル評価・選択")

        st.subheader("生成タイトル")
        for i, gen_title in enumerate(st.session_state.generated_titles):
            cols = st.columns([0.5, 2, 2, 1, 1, 2, 1])  # 評価コメントと修正用カラムを追加
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
                            # refined_main = refined_title.get("main_title", "")
                            # refined_sub = refined_title.get("sub_title", "")

                            # 修正後
                            refined_main = refined_title.main_title
                            refined_sub = refined_title.sub_title

                            full_refined_title = f"{refined_main} - {refined_sub}"
                            analysis = st.session_state.evaluator.evaluate_title(full_refined_title, category)
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
                            # st.experimental_rerun()
                            st.rerun()  # 新しい書き方

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
                            analysis = st.session_state.evaluator.evaluate_title(
                                full_title,
                                st.session_state.selected_category
                            )
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

        # Step 3: 見出し生成
        if st.session_state.generated_titles:
            st.header("Step 3: 見出し生成")

            available_titles = []
            for gen_title in st.session_state.generated_titles:
                full_title = f"{gen_title.main_title} - {gen_title.sub_title}"
                available_titles.append(full_title)

            st.session_state.selected_title_for_headline = st.selectbox(
                "見出しを生成するタイトルを選択してください",
                options=available_titles
            )

            with st.expander("見出し生成プロンプトの編集", expanded=False):
                st.session_state.headline_prompt = st.text_area(
                    "プロンプトテンプレート",
                    value=st.session_state.headline_prompt,
                    height=400
                )

            if st.button("見出しを生成", key="generate_headlines"):
                with st.spinner("見出しを生成中..."):
                    try:
                        headlines = headline_generator.generate_headlines(
                            st.session_state.selected_title_for_headline,
                            st.session_state.target_audience, # ターゲット像を引数に追加
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

                # Step 4: 本文生成
                st.header("Step 4: 本文生成")

                with st.expander("本文生成プロンプトの編集", expanded=False):
                    st.session_state.body_prompt = st.text_area(
                        "本文生成プロンプトテンプレート",
                        value=st.session_state.body_prompt,
                        height=400
                    )

                if st.button("本文を生成", key="generate_body"):
                    with st.spinner("本文を生成中..."):
                        try:
                            st.session_state.generated_body = body_generator.generate_body(
                                st.session_state.selected_title_for_headline,
                                st.session_state.manual_headlines,
                                st.session_state.target_audience, # ターゲット像を引数に追加
                                st.session_state.body_prompt
                            )
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")

                if st.session_state.generated_body:
                    st.subheader("生成された本文")
                    st.write(st.session_state.generated_body)

if __name__ == "__main__":
    main()
