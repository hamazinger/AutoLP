import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json
import openai
import requests
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract
from PyPDF2 import PdfReader
from docx import Document
from google.cloud import bigquery
from google.oauth2 import service_account

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

class TitleGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.url_extractor = URLContentExtractor()
        # 初期テンプレート（ユーザーがUIで編集可能）
        self.user_editable_prompt = """
あなたはセミナータイトルの生成を行うプロフェッショナルなコピーライターです。以下の制約条件と入力された情報をもとにセミナータイトルを生成してください。

# 制約条件
- メインタイトルとサブタイトルに分ける
- メインタイトルでは、問題点や課題、悩み、不安を投げかける
- サブタイトルでは、メインタイトルで表現したインサイトを解決する手段や手法、アプローチ、その先に得られるベネフィットを表現する
- メインタイトル、サブタイトルは、それぞれ40文字以内で簡潔にする
- 感嘆符（！）は使用しない
- 参加したら何がわかるのかが明確である
"""
        # JSONで出力させるための固定指示文
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

    def _create_prompt(
        self, 
        context: str, 
        base_prompt: str, 
        revision_instructions: str, 
        product_url: str, 
        file_content: str
    ) -> str:
        """タイトル生成用の最終プロンプトを作るヘルパー関数"""
        additional_context = ""
        if product_url:
            content = self.url_extractor.extract_with_trafilatura(product_url)
            if content and not content.error:
                additional_context += f"\n製品タイトル: {content.title}\n製品説明: {content.description}\n製品詳細: {content.main_content[:1000]}\n"
            else:
                st.warning(f"製品情報の取得に失敗しました: {content.error if content else '不明なエラー'}")
        if file_content:
            additional_context += f"\nアップロードファイルの内容:\n{file_content}"
        
        prompt = f"""
# 入力情報
{context}
{additional_context}

{base_prompt}
"""
        if revision_instructions:
            prompt += f"\n## 修正指示\n{revision_instructions}\n"
        
        prompt += self.fixed_output_instructions
        return prompt

    def generate_titles(
        self, 
        context: str, 
        prompt_template: str = None, 
        product_url: str = None, 
        file_content: str = None,
        revision_instructions: str = None
    ) -> List[Dict[str, str]]:
        """
        revision_instructions: タイトル再生成時の追加修正指示
        """
        base_prompt = prompt_template or self.user_editable_prompt
        final_prompt = self._create_prompt(
            context=context,
            base_prompt=base_prompt,
            revision_instructions=revision_instructions or "",
            product_url=product_url or "",
            file_content=file_content or ""
        )
        result_text = ""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0
            )
            
            result_text = response.choices[0].message['content'].strip()
            
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
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}\nAIからの応答:\n{result_text}")
            return []

    def revise_single_title(
        self,
        original_main: str,
        original_sub: str,
        revision_instructions: str
    ) -> Dict[str, str]:
        """
        1つのタイトル（メイン＆サブ）に対して修正指示を反映し、再生成するメソッド。
        JSONで { "main_title": "...", "sub_title": "..." } の形式を返す。
        """
        prompt = f"""
あなたは優秀なコピーライターです。
以下の「元のタイトル」をリライト・修正してください。

元のメインタイトル: {original_main}
元のサブタイトル: {original_sub}

# 修正指示
{revision_instructions}

修正後はJSON形式で出力してください。
フォーマット:
{{
    "main_title": "修正後のメインタイトル",
    "sub_title": "修正後のサブタイトル"
}}
"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            result_text = response.choices[0].message['content'].strip()
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                if start_index != -1 and end_index > start_index:
                    json_text = result_text[start_index:end_index]
                    result = json.loads(json_text)
                else:
                    st.error("修正後タイトルを抽出できませんでした。")
                    return {"main_title": original_main, "sub_title": original_sub}
            
            return result
        except Exception as e:
            st.error(f"OpenAI APIエラー: {str(e)}")
            return {"main_title": original_main, "sub_title": original_sub}


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
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.fixed_prompt_part = """
「『{title}』というタイトルのイベントを企画しており、その告知文を作成します。 告知文を作成する前に、以下の内容でその見出しを３つ作成してください。それぞれの見出しは簡潔な文章としてください。 」
"""
        # 初期テンプレート（ユーザーがUIで編集可能）
        self.user_editable_prompt = """
見出し1：このセミナーを開催する、社会や企業の背景
見出し2：このセミナーで訴求したい、課題、問題、悩み、不安
見出し3：上記課題の解決の方向性
"""
        # JSONで出力させるための固定指示文
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含まれず、JSONオブジェクトのみを出力してください：
{
    "background": "背景の見出し",
    "problem": "課題の見出し",
    "solution": "解決策の見出し"
}
"""

    def generate_headlines(
        self, 
        title: str, 
        prompt_template: str = None, 
        revision_instructions: str = None
    ) -> HeadlineSet:
        base_prompt = self.fixed_prompt_part.format(title=title)
        
        user_prompt_part = prompt_template or self.user_editable_prompt
        if revision_instructions:
            user_prompt_part += f"\n## 修正指示\n{revision_instructions}\n"
        
        prompt = base_prompt + user_prompt_part + self.fixed_output_instructions
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            result_text = response.choices[0].message['content'].strip()
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                if start_index != -1 and end_index > start_index:
                    json_text = result_text[start_index:end_index]
                    result = json.loads(json_text)
                else:
                    return HeadlineSet("", "", "")
            
            return HeadlineSet.from_dict(result)
            
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return HeadlineSet("", "", "")

    def revise_single_headline(
        self,
        original_text: str,
        revision_instructions: str,
        headline_type: str
    ) -> str:
        """
        1つの見出し（背景 or 課題 or 解決策）に対して修正指示を反映し、再生成するメソッド。
        headline_type: '背景', '課題', '解決策' など
        JSONで {"new_text": "..."} の形式を返す想定。
        """
        prompt = f"""
あなたは優秀なコピーライターです。
以下の「元の{headline_type}見出し」を修正してください。

元の{headline_type}見出し: {original_text}

# 修正指示
{revision_instructions}

修正後の{headline_type}見出しをJSON形式で出力してください。
フォーマット:
{{
    "new_text": "修正後の見出し"
}}
"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            result_text = response.choices[0].message['content'].strip()
            try:
                result = json.loads(result_text)
                return result.get("new_text", original_text)
            except json.JSONDecodeError:
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                if start_index != -1 and end_index > start_index:
                    json_text = result_text[start_index:end_index]
                    result = json.loads(json_text)
                    return result.get("new_text", original_text)
                else:
                    st.error("修正後見出しを抽出できませんでした。")
                    return original_text
        except Exception as e:
            st.error(f"OpenAI APIエラー: {str(e)}")
            return original_text

class BodyGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.fixed_prompt_part = """
以下のセミナータイトルと見出しに基づいて、本文を生成してください：
- 各見出しは本文中に明示してください。明確に見出しであることがわかるマークダウンの書式（見出しレベル4）を用いてください。

タイトル：「{title}」
【背景】:{background}
【課題】:{problem}
【解決策】:{solution}
"""
        # 初期テンプレート（ユーザーがUIで編集可能）
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
"""

    def generate_body(
        self, 
        title: str, 
        headlines: HeadlineSet, 
        prompt_template: str = None,
        revision_instructions: str = None
    ) -> str:
        base_prompt = self.fixed_prompt_part.format(
            title=title,
            background=headlines.background,
            problem=headlines.problem,
            solution=headlines.solution
        )
        
        user_prompt_part = prompt_template or self.user_editable_prompt
        if revision_instructions:
            user_prompt_part += f"\n## 修正指示\n{revision_instructions}\n"
        
        prompt = base_prompt + user_prompt_part
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            return response.choices[0].message['content'].strip()
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return ""

    def revise_single_body_section(
        self,
        original_text: str,
        revision_instructions: str,
        section_type: str
    ) -> str:
        """
        1つの本文セクション（背景・課題・解決策）に対して修正指示を反映するメソッド。
        section_type: '背景', '課題', '解決策' など
        JSONで {"new_text": "..."} の形式を返す想定。
        """
        prompt = f"""
あなたは優秀なコピーライターです。
以下の「{section_type}」本文を修正してください。

元の{section_type}本文:
{original_text}

# 修正指示
{revision_instructions}

修正後の{section_type}本文をJSON形式で出力してください。
フォーマット:
{{
    "new_text": "修正後の本文"
}}
"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            result_text = response.choices[0].message['content'].strip()
            try:
                result = json.loads(result_text)
                return result.get("new_text", original_text)
            except json.JSONDecodeError:
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                if start_index != -1 and end_index > start_index:
                    json_text = result_text[start_index:end_index]
                    result = json.loads(json_text)
                    return result.get("new_text", original_text)
                else:
                    st.error("修正後本文を抽出できませんでした。")
                    return original_text
        except Exception as e:
            st.error(f"OpenAI APIエラー: {str(e)}")
            return original_text


class InMemoryCache:
    def __init__(self):
        if 'title_cache' not in st.session_state:
            st.session_state.title_cache = {}
    
    def get_evaluation(self, title: str) -> Optional[TitleEvaluation]:
        return st.session_state.title_cache.get(title)
    
    def set_evaluation(self, title: str, evaluation: TitleEvaluation):
        st.session_state.title_cache[title] = evaluation

def init_bigquery_client():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return bigquery.Client(credentials=credentials)
    except Exception as e:
        st.warning("BigQueryの認証情報が設定されていないか、読み込みに失敗しました。BigQueryデータは使用しません。")
        return None

def load_seminar_data():
    client = init_bigquery_client()
    if client is None:
        return None
    
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
        st.warning(f"BigQueryからのデータ読み込みに失敗しました: {str(e)}")
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

    # 追加修正指示を保存するState
    if 'title_revision_instructions' not in st.session_state:
        st.session_state.title_revision_instructions = ""
    if 'headline_revision_instructions' not in st.session_state:
        st.session_state.headline_revision_instructions = ""
    if 'body_revision_instructions' not in st.session_state:
        st.session_state.body_revision_instructions = ""

    # 生成した本文を3セクションに分割して保持する
    if 'body_section_background' not in st.session_state:
        st.session_state.body_section_background = ""
    if 'body_section_problem' not in st.session_state:
        st.session_state.body_section_problem = ""
    if 'body_section_solution' not in st.session_state:
        st.session_state.body_section_solution = ""

def main():
    init_session_state()
    
    # OpenAIのAPIキーを読み込み
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.warning("OpenAI APIキーが設定されていません。制限された動作となります。")
        openai.api_key = "sk-xxx"  # ダミー
    
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
                    st.error(f"カテゴリデータの処理中にエラーが発生しました: {str(e)}")
            else:
                st.info("セミナーデータを利用しません。")
    
    model_name = "gpt-3.5-turbo"
    title_generator = TitleGenerator(openai.api_key, model=model_name)
    headline_generator = HeadlineGenerator(openai.api_key, model=model_name)
    body_generator = BodyGenerator(openai.api_key, model=model_name)
    cache = InMemoryCache()
    
    st.header("Step 1: 基本情報入力")
    
    col1, col2, col3 = st.columns([1, 1, 1])
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
        if st.session_state.available_categories:
            category = st.selectbox(
                "カテゴリ",
                options=st.session_state.available_categories
            )
            st.session_state.selected_category = category
        else:
            st.session_state.selected_category = None
            st.write("カテゴリデータはありません")

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
            st.error(f"ファイルの読み込みでエラーが発生しました: {str(e)}")
    
    with st.expander("タイトル生成プロンプトの編集", expanded=False):
        st.session_state.title_prompt = st.text_area(
            "プロンプトテンプレート",
            value=st.session_state.title_prompt,
            height=300
        )
    
    if st.button("タイトルを生成", key="generate_titles"):
        st.session_state.title_revision_instructions = ""  # 新規生成のタイミングで修正指示をリセット
        context = f"""
ペインポイント: {pain_points}
カテゴリ: {st.session_state.selected_category}
"""
        with st.spinner("タイトルを生成中..."):
            try:
                titles = title_generator.generate_titles(
                    context,
                    st.session_state.title_prompt,
                    product_url,
                    file_content,
                    revision_instructions=None
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
                        analysis = st.session_state.evaluator.evaluate_title(full_title, st.session_state.selected_category)
                        evaluation = TitleEvaluation(
                            speed=analysis.predicted_speed,
                            grade=analysis.grade,
                            comment=analysis.evaluation_comment
                        )
                        cache.set_evaluation(full_title, evaluation)
                    st.session_state.generated_titles.append(
                        GeneratedTitle(main_title=main_title, sub_title=sub_title, evaluation=evaluation)
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

    # タイトル一覧 & 個別修正
    if st.session_state.generated_titles:
        st.header("Step 2: 生成されたタイトルの評価・修正")

        for i, gen_title in enumerate(st.session_state.generated_titles):
            with st.expander(f"タイトル候補 #{i+1}", expanded=False):
                full_title = f"{gen_title.main_title} - {gen_title.sub_title}"
                st.write(f"**メインタイトル:** {gen_title.main_title}")
                st.write(f"**サブタイトル:** {gen_title.sub_title}")
                st.metric("集客速度", f"{gen_title.evaluation.speed:.1f}")
                grade_colors = {"A": "green", "B": "orange", "C": "red"}
                grade_color = grade_colors.get(gen_title.evaluation.grade, "gray")
                st.markdown(
                    f'<p style="color: {grade_color}; font-weight: bold;">評価: {gen_title.evaluation.grade}</p>',
                    unsafe_allow_html=True
                )
                st.write(f"**評価コメント:** {gen_title.evaluation.comment}")

                # 評価詳細表示
                display_button = st.checkbox(f"詳細評価を見る (#{i+1})", key=f"detail_{i}_check")
                if display_button:
                    display_evaluation_details(full_title, st.session_state.evaluator)

                # 個別タイトル修正
                revision_text_single = st.text_area(
                    f"このタイトル(#{i+1})への修正指示",
                    key=f"revise_instruction_single_title_{i}",
                    height=80
                )
                if st.button(f"このタイトルだけ再生成 (#{i+1})", key=f"revise_single_title_{i}"):
                    with st.spinner("タイトルを修正・再生成中..."):
                        revised = title_generator.revise_single_title(
                            original_main=gen_title.main_title,
                            original_sub=gen_title.sub_title,
                            revision_instructions=revision_text_single
                        )
                        # 再生成したタイトルに対して再評価
                        new_main = revised.get("main_title", gen_title.main_title)
                        new_sub = revised.get("sub_title", gen_title.sub_title)
                        new_full = f"{new_main} - {new_sub}"
                        analysis = st.session_state.evaluator.evaluate_title(
                            new_full, st.session_state.selected_category
                        )
                        new_eval = TitleEvaluation(
                            speed=analysis.predicted_speed,
                            grade=analysis.grade,
                            comment=analysis.evaluation_comment
                        )
                        cache.set_evaluation(new_full, new_eval)
                        # 置き換え
                        st.session_state.generated_titles[i] = GeneratedTitle(new_main, new_sub, new_eval)
                        st.success("タイトルを修正しました！")

        # 手動タイトル評価
        st.subheader("手動タイトル評価の追加")
        col_a, col_b = st.columns([4, 1])
        with col_a:
            manual_main_title = st.text_input("メインタイトル (手動)", key="manual_main_title_input")
            manual_sub_title = st.text_input("サブタイトル (手動)", key="manual_sub_title_input")
        with col_b:
            if st.button("評価する", key="evaluate_manual"):
                if not manual_main_title:
                    st.warning("メインタイトルを入力してください。")
                else:
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

                            # 末尾に追加
                            st.session_state.generated_titles.append(
                                GeneratedTitle(
                                    main_title=manual_main_title,
                                    sub_title=manual_sub_title,
                                    evaluation=evaluation
                                )
                            )
                            display_evaluation_details(full_title, st.session_state.evaluator)
                        except Exception as e:
                            st.error(f"エラーが発生しました: {str(e)}")

    # Step 3: 見出し生成・個別修正
    if st.session_state.generated_titles:
        st.header("Step 3: 見出し生成")
        # タイトル選択
        available_titles = []
        for gen_title in st.session_state.generated_titles:
            full_t = f"{gen_title.main_title} - {gen_title.sub_title}"
            available_titles.append(full_t)
        if len(available_titles) > 0:
            st.session_state.selected_title_for_headline = st.selectbox(
                "見出しを生成するタイトルを選択してください",
                options=available_titles
            )

            with st.expander("見出し生成プロンプトの編集", expanded=False):
                st.session_state.headline_prompt = st.text_area(
                    "見出しプロンプトテンプレート",
                    value=st.session_state.headline_prompt,
                    height=300
                )

            if st.button("見出しを生成", key="generate_headlines"):
                st.session_state.headline_revision_instructions = ""  # 新規生成でリセット
                with st.spinner("見出しを生成中..."):
                    try:
                        headlines = headline_generator.generate_headlines(
                            st.session_state.selected_title_for_headline,
                            st.session_state.headline_prompt,
                            revision_instructions=None
                        )
                        st.session_state.headlines = headlines
                        st.session_state.manual_headlines = headlines
                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")

            if st.session_state.manual_headlines:
                st.subheader("生成された見出し（個別修正可能）")

                # 背景
                st.write("#### 背景")
                background_text = st.text_area(
                    "背景の見出し",
                    value=st.session_state.manual_headlines.background,
                    key="edit_background_textarea_headline",
                    height=100
                )
                revise_bg_headline = st.text_area("背景への修正指示 (見出し)", key="revise_background_instructions_headline", height=60)
                if st.button("この背景だけ再生成 (見出し)", key="revise_background_btn_headline"):
                    with st.spinner("背景見出しを修正中..."):
                        new_bg = headline_generator.revise_single_headline(
                            original_text=background_text,
                            revision_instructions=revise_bg_headline,
                            headline_type="背景"
                        )
                        st.session_state.manual_headlines.background = new_bg
                        st.success("背景見出しを修正しました！")

                # 課題
                st.write("#### 課題")
                problem_text = st.text_area(
                    "課題の見出し",
                    value=st.session_state.manual_headlines.problem,
                    key="edit_problem_textarea_headline",
                    height=100
                )
                revise_pb_headline = st.text_area("課題への修正指示 (見出し)", key="revise_problem_instructions_headline", height=60)
                if st.button("この課題だけ再生成 (見出し)", key="revise_problem_btn_headline"):
                    with st.spinner("課題見出しを修正中..."):
                        new_pb = headline_generator.revise_single_headline(
                            original_text=problem_text,
                            revision_instructions=revise_pb_headline,
                            headline_type="課題"
                        )
                        st.session_state.manual_headlines.problem = new_pb
                        st.success("課題見出しを修正しました！")

                # 解決策
                st.write("#### 解決策")
                solution_text = st.text_area(
                    "解決策の見出し",
                    value=st.session_state.manual_headlines.solution,
                    key="edit_solution_textarea_headline",
                    height=100
                )
                revise_sol_headline = st.text_area("解決策への修正指示 (見出し)", key="revise_solution_instructions_headline", height=60)
                if st.button("この解決策だけ再生成 (見出し)", key="revise_solution_btn_headline"):
                    with st.spinner("解決策見出しを修正中..."):
                        new_sol = headline_generator.revise_single_headline(
                            original_text=solution_text,
                            revision_instructions=revise_sol_headline,
                            headline_type="解決策"
                        )
                        st.session_state.manual_headlines.solution = new_sol
                        st.success("解決策見出しを修正しました！")

                # Step 4: 本文生成
                st.header("Step 4: 本文生成")
                with st.expander("本文生成プロンプトの編集", expanded=False):
                    st.session_state.body_prompt = st.text_area(
                        "本文生成プロンプトテンプレート",
                        value=st.session_state.body_prompt,
                        height=300
                    )

                if st.button("本文を生成", key="generate_body"):
                    st.session_state.body_revision_instructions = ""
                    with st.spinner("本文を生成中..."):
                        try:
                            generated = body_generator.generate_body(
                                st.session_state.selected_title_for_headline,
                                st.session_state.manual_headlines,
                                st.session_state.body_prompt,
                                revision_instructions=None
                            )
                            st.session_state.generated_body = generated
                        except Exception as e:
                            st.error(f"エラーが発生しました: {str(e)}")

                # 本文をセクション分割して表示 & 個別修正
                if st.session_state.generated_body:
                    st.subheader("生成された本文（セクションごとに修正可能）")

                    # ここでは簡易的に「#### 」区切りで3セクションに分ける想定
                    sections = st.session_state.generated_body.split("#### ")
                    while len(sections) < 4:
                        sections.append("")  # セクション不足なら補填

                    # [1],[2],[3]を背景/課題/解決策とみなす
                    background_section = sections[1] if len(sections) > 1 else ""
                    problem_section = sections[2] if len(sections) > 2 else ""
                    solution_section = sections[3] if len(sections) > 3 else ""

                    # 背景
                    st.write("### 背景セクション")
                    st.session_state.body_section_background = st.text_area(
                        "背景本文",
                        value=background_section,
                        key="body_bg_textarea_body",
                        height=200
                    )
                    revise_bg_body = st.text_area("背景本文への修正指示 (本文)", key="revise_bg_body_instructions_body", height=60)
                    if st.button("この背景本文だけ再生成", key="revise_bg_body_btn"):
                        with st.spinner("背景本文を修正中..."):
                            new_bg_body = body_generator.revise_single_body_section(
                                original_text=st.session_state.body_section_background,
                                revision_instructions=revise_bg_body,
                                section_type="背景"
                            )
                            st.session_state.body_section_background = new_bg_body
                            st.success("背景本文を修正しました！")

                    # 課題
                    st.write("### 課題セクション")
                    st.session_state.body_section_problem = st.text_area(
                        "課題本文",
                        value=problem_section,
                        key="body_pb_textarea_body",
                        height=200
                    )
                    revise_pb_body = st.text_area("課題本文への修正指示 (本文)", key="revise_pb_body_instructions_body", height=60)
                    if st.button("この課題本文だけ再生成", key="revise_pb_body_btn"):
                        with st.spinner("課題本文を修正中..."):
                            new_pb_body = body_generator.revise_single_body_section(
                                original_text=st.session_state.body_section_problem,
                                revision_instructions=revise_pb_body,
                                section_type="課題"
                            )
                            st.session_state.body_section_problem = new_pb_body
                            st.success("課題本文を修正しました！")

                    # 解決策
                    st.write("### 解決策セクション")
                    st.session_state.body_section_solution = st.text_area(
                        "解決策本文",
                        value=solution_section,
                        key="body_sol_textarea_body",
                        height=200
                    )
                    revise_sol_body = st.text_area("解決策本文への修正指示 (本文)", key="revise_sol_body_instructions_body", height=60)
                    if st.button("この解決策本文だけ再生成", key="revise_sol_body_btn"):
                        with st.spinner("解決策本文を修正中..."):
                            new_sol_body = body_generator.revise_single_body_section(
                                original_text=st.session_state.body_section_solution,
                                revision_instructions=revise_sol_body,
                                section_type="解決策"
                            )
                            st.session_state.body_section_solution = new_sol_body
                            st.success("解決策本文を修正しました！")

                    # 再構築ボタン
                    if st.button("修正済みセクションを統合して本文を表示", key="integrate_sections_btn"):
                        new_full_body = (
                            f"#### {st.session_state.body_section_background}\n\n"
                            f"#### {st.session_state.body_section_problem}\n\n"
                            f"#### {st.session_state.body_section_solution}"
                        )
                        st.session_state.generated_body = new_full_body
                        st.subheader("最終的な本文")
                        st.write(new_full_body)

if __name__ == "__main__":
    main()
