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

# Streamlitのページ設定
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

# 修正履歴付きタイトルクラス
class GeneratedTitleWithRevision:
    def __init__(self, main_title: str, sub_title: str, evaluation: TitleEvaluation):
        self.main_title = main_title
        self.sub_title = sub_title
        self.evaluation = evaluation
        self.revision_history = []

    def add_revision(self, revision_request: str, revised_main: str, revised_sub: str):
        self.revision_history.append({
            'request': revision_request,
            'original_main': self.main_title,
            'original_sub': self.sub_title,
            'revised_main': revised_main,
            'revised_sub': revised_sub,
            'timestamp': datetime.now().isoformat()
        })
        self.main_title = revised_main
        self.sub_title = revised_sub

# 見出し修正履歴クラス
class HeadlineRevision:
    def __init__(self, headline: HeadlineSet):
        self.current = headline
        self.revision_history = []

    def add_revision(self, revision_request: str, revised_headline: HeadlineSet):
        self.revision_history.append({
            'request': revision_request,
            'original': self.current,
            'revised': revised_headline,
            'timestamp': datetime.now().isoformat()
        })
        self.current = revised_headline

# 本文修正履歴クラス
class BodyRevision:
    def __init__(self, body: str):
        self.current = body
        self.revision_history = []

    def add_revision(self, revision_request: str, revised_body: str):
        self.revision_history.append({
            'request': revision_request,
            'original': self.current,
            'revised': revised_body,
            'timestamp': datetime.now().isoformat()
        })
        self.current = revised_body

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
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        openai.api_key = api_key
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
"""
        self.revision_prompt_template = """
前回生成したタイトルに対して、以下の修正要望が来ています：
{revision_request}

前回生成したタイトル：
メインタイトル：{main_title}
サブタイトル：{sub_title}

この修正要望を踏まえて、タイトルを改善してください。なお、先ほどの制約条件は依然として有効です。
"""
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください：
{
    "titles": [
        {
            "main_title": "メインタイトル1",
            "sub_title": "サブタイトル1"
        }
    ]
}
"""

    def generate_titles(self, context: str, prompt_template: str = None, product_url: str = None, file_content: str = None) -> List[Dict[str, str]]:
        additional_context = self._prepare_additional_context(product_url, file_content)
        prompt = self._build_prompt(context, additional_context, prompt_template)
        return self._execute_generation(prompt, num_titles=3)

    def revise_title(self, original_title: Dict[str, str], revision_request: str) -> Dict[str, str]:
        revision_prompt = self._build_revision_prompt(original_title, revision_request)
        results = self._execute_generation(revision_prompt, num_titles=1)
        return results[0] if results else None

    def _prepare_additional_context(self, product_url: str = None, file_content: str = None) -> str:
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
        return additional_context

    def _build_prompt(self, context: str, additional_context: str, prompt_template: str = None) -> str:
        return f"""
# 入力情報
{context}
{additional_context}
""" + (prompt_template or self.user_editable_prompt) + self.fixed_output_instructions

    def _build_revision_prompt(self, original_title: Dict[str, str], revision_request: str) -> str:
        return self.revision_prompt_template.format(
            revision_request=revision_request,
            main_title=original_title["main_title"],
            sub_title=original_title["sub_title"]
        ) + self.fixed_output_instructions

    def _execute_generation(self, prompt: str, num_titles: int = 1) -> List[Dict[str, str]]:
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
                    raise ValueError("タイトルを抽出できませんでした")
    
            if not isinstance(result, dict) or "titles" not in result:
                raise ValueError("不正な応答形式です")
            
            titles = result["titles"]
            if not isinstance(titles, list) or not titles:
                raise ValueError("タイトルが見つかりません")
            
            return titles[:num_titles]
                
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return []

class HeadlineGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        openai.api_key = api_key
        self.model = model
        self.fixed_prompt_part = """
『{title}』というタイトルのイベントを企画しており、その告知文を作成します。告知文を作成する前に、以下の内容でその見出しを作成してください。見出しは簡潔な文章としてください。
"""
        self.user_editable_prompt = """
見出し1：このセミナーを開催する、社会や企業の背景
見出し2：このセミナーで訴求したい、課題、問題、悩み、不安
見出し3：上記課題の解決の方向性
"""
        self.revision_prompt_template = """
前回生成した見出しに対して、以下の修正要望が来ています：
{revision_request}

前回生成した見出し：
背景：{background}
課題：{problem}
解決策：{solution}

この修正要望を踏まえて、見出しを改善してください。
"""
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください：
{
    "background": "背景の見出し",
    "problem": "課題の見出し",
    "solution": "解決策の見出し"
}
"""

    def generate_headlines(self, title: str, prompt_template: str = None) -> HeadlineSet:
        """タイトルに基づいて見出しを生成"""
        prompt = self._build_prompt(title, prompt_template)
        return self._execute_generation(prompt)

    def revise_headline(self, headlines: HeadlineSet, revision_request: str) -> HeadlineSet:
        """修正要望に基づいて見出しを修正"""
        revision_prompt = self._build_revision_prompt(headlines, revision_request)
        return self._execute_generation(revision_prompt)

    def _build_prompt(self, title: str, prompt_template: str = None) -> str:
        return self.fixed_prompt_part.format(title=title) + (prompt_template or self.user_editable_prompt) + self.fixed_output_instructions

    def _build_revision_prompt(self, headlines: HeadlineSet, revision_request: str) -> str:
        return self.revision_prompt_template.format(
            revision_request=revision_request,
            background=headlines.background,
            problem=headlines.problem,
            solution=headlines.solution
        ) + self.fixed_output_instructions

    def _execute_generation(self, prompt: str) -> HeadlineSet:
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
            
            return HeadlineSet.from_dict(result)
            
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return HeadlineSet("", "", "")

class BodyGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        openai.api_key = api_key
        self.model = model
        self.fixed_prompt_part = """
以下のセミナータイトルと見出しに基づいて、本文を生成してください：
- 各見出しは本文中に明示してください。明確に見出しであることがわかるマークダウンの書式（見出しレベル4）を用いてください。

タイトル：「{title}」
背景：{background}
課題：{problem}
解決策：{solution}
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
"""
        self.revision_prompt_template = """
現在の本文に対して、以下の修正要望が来ています：
{revision_request}

前回の本文：
{previous_body}

この修正要望を踏まえて、本文を改善してください。なお、先ほどの制約条件は依然として有効です。
"""

    def generate_body(self, title: str, headlines: HeadlineSet, prompt_template: str = None) -> str:
        """タイトルと見出しに基づいて本文を生成"""
        prompt = self._build_prompt(title, headlines, prompt_template)
        return self._execute_generation(prompt)

    def revise_body(self, title: str, headlines: HeadlineSet, previous_body: str, revision_request: str) -> str:
        """修正要望に基づいて本文を修正"""
        revision_prompt = self._build_revision_prompt(title, headlines, previous_body, revision_request)
        return self._execute_generation(revision_prompt)

    def _build_prompt(self, title: str, headlines: HeadlineSet, prompt_template: str = None) -> str:
        return self.fixed_prompt_part.format(
            title=title,
            background=headlines.background,
            problem=headlines.problem,
            solution=headlines.solution
        ) + (prompt_template or self.user_editable_prompt)

    def _build_revision_prompt(self, title: str, headlines: HeadlineSet, previous_body: str, revision_request: str) -> str:
        context = self._build_prompt(title, headlines, None)
        return context + self.revision_prompt_template.format(
            revision_request=revision_request,
            previous_body=previous_body
        )

    def _execute_generation(self, prompt: str) -> str:
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
    """セッション状態の初期化"""
    initial_states = {
        'generated_titles': [],
        'selected_title': None,
        'selected_title_for_headline': None,
        'selected_category': None,
        'headlines': None,
        'title_cache': {},
        'seminar_data': None,
        'evaluator': None,
        'available_categories': [],
        'extracted_content': {},
        'title_prompt': TitleGenerator("dummy_key").user_editable_prompt,
        'headline_prompt': HeadlineGenerator("dummy_key").user_editable_prompt,
        'body_prompt': BodyGenerator("dummy_key").user_editable_prompt,
        'generated_body': None,
        'manual_headlines': None,
        'title_revisions': {},
        'headline_revisions': {},
        'body_revisions': {},
    }
    
    for key, initial_value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = initial_value

def display_generated_title(i: int, title: GeneratedTitleWithRevision, title_generator, category, evaluator):
    """生成されたタイトルとその修正UIを表示"""
    cols = st.columns([0.5, 2, 2, 1, 1, 2])
    with cols[0]:
        if st.radio(
            "選択",
            ["✓"],
            key=f"radio_{i}",
            label_visibility="collapsed"
        ):
            st.session_state.selected_title = f"{title.main_title} - {title.sub_title}"
    with cols[1]:
        st.write(f"**メインタイトル:** {title.main_title}")
    with cols[2]:
        st.write(f"**サブタイトル:** {title.sub_title}")
    with cols[3]:
        st.metric("集客速度", f"{title.evaluation.speed:.1f}")
    with cols[4]:
        grade_colors = {"A": "green", "B": "orange", "C": "red"}
        grade_color = grade_colors.get(title.evaluation.grade, "gray")
        st.markdown(
            f'<p style="color: {grade_color}; font-weight: bold; text-align: center;">評価: {title.evaluation.grade}</p>',
            unsafe_allow_html=True
        )
    with cols[5]:
        st.write(f"**評価:** {title.evaluation.comment}")

    # 個別の修正セクション
    with st.expander(f"このタイトルを修正", expanded=False):
        revision_request = st.text_area(
            "修正要望を入力してください",
            key=f"title_revision_{i}",
            help="例: もっと具体的な課題を入れて、解決策をより明確に"
        )
        if st.button("修正する", key=f"revise_title_{i}"):
            with st.spinner("タイトルを修正中..."):
                try:
                    revised_title = title_generator.revise_title(
                        {"main_title": title.main_title, "sub_title": title.sub_title},
                        revision_request
                    )
                    if revised_title:
                        title.add_revision(
                            revision_request,
                            revised_title["main_title"],
                            revised_title["sub_title"]
                        )
                        
                        # 評価の更新
                        full_title = f"{revised_title['main_title']} - {revised_title['sub_title']}"
                        analysis = evaluator.evaluate_title(full_title, category)
                        title.evaluation = TitleEvaluation(
                            speed=analysis.predicted_speed,
                            grade=analysis.grade,
                            comment=analysis.evaluation_comment
                        )
                        st.success("タイトルを修正しました！")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

        # 修正履歴の表示
        if title.revision_history:
            st.write("#### 修正履歴")
            for idx, revision in enumerate(title.revision_history, 1):
                with st.expander(f"修正 {idx}"):
                    st.write(f"**修正要望:** {revision['request']}")
                    st.write(f"**修正前:** {revision['original_main']} - {revision['original_sub']}")
                    st.write(f"**修正後:** {revision['revised_main']} - {revision['revised_sub']}")

def display_headline_section(headline: HeadlineSet, section_key: str):
    """見出しセクションとその修正UIを表示"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_area(
            section_key.capitalize(),
            value=getattr(headline, section_key.lower()),
            key=f"edit_{section_key}",
            disabled=True
        )
    with col2:
        revision_request = st.text_area(
            "修正要望",
            key=f"headline_revision_{section_key}",
            help="例: より具体的に、明確に"
        )
        return revision_request

def main():
    init_session_state()
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OpenAI APIキーが設定されていません")
        return

    st.title("セミナータイトルジェネレーター")

    # データ読み込み
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
                    return
            else:
                st.error("データの読み込みに失敗しました。")
                return

    # ジェネレーターの初期化
    model_name = "gpt-4o"
    title_generator = TitleGenerator(api_key, model=model_name)
    headline_generator = HeadlineGenerator(api_key, model=model_name)
    body_generator = BodyGenerator(api_key, model=model_name)
    cache = InMemoryCache()

    st.header("Step 1: 基本情報入力")
    
    # 入力フォーム
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
        category = st.selectbox(
            "カテゴリ",
            options=st.session_state.available_categories
        )
        st.session_state.selected_category = category

    # ファイルアップロード
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

    # タイトル生成セクション
    with st.expander("タイトル生成プロンプトの編集", expanded=False):
        st.session_state.title_prompt = st.text_area(
            "プロンプトテンプレート",
            value=st.session_state.title_prompt,
            height=400
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
                    st.session_state.title_prompt,
                    product_url,
                    file_content
                )
                st.session_state.generated_titles = []
                for title_data in titles:
                    main_title = title_data.get("main_title", "")
                    sub_title = title_data.get("sub_title", "")
                    full_title = f"{main_title} - {sub_title}"
                    
                    analysis = st.session_state.evaluator.evaluate_title(full_title, category)
                    evaluation = TitleEvaluation(
                        speed=analysis.predicted_speed,
                        grade=analysis.grade,
                        comment=analysis.evaluation_comment
                    )
                    
                    title_with_revision = GeneratedTitleWithRevision(main_title, sub_title, evaluation)
                    st.session_state.generated_titles.append(title_with_revision)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

    # タイトル表示・修正セクション
    if st.session_state.generated_titles:
        st.header("Step 2: タイトル評価・選択")
        
        for i, title in enumerate(st.session_state.generated_titles):
            display_generated_title(i, title, title_generator, category, st.session_state.evaluator)

        # 見出し生成セクション
        if st.session_state.selected_title:
            st.header("Step 3: 見出し生成")
            
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
                            st.session_state.selected_title,
                            st.session_state.headline_prompt
                        )
                        st.session_state.headlines = HeadlineRevision(headlines)
                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")

            # 見出し表示・修正セクション
            if st.session_state.headlines:
                st.subheader("生成された見出し")
                headlines = st.session_state.headlines.current
                
                background_request = display_headline_section(headlines, "background")
                problem_request = display_headline_section(headlines, "problem")
                solution_request = display_headline_section(headlines, "solution")
                
                # 見出しの個別修正ボタン
                cols = st.columns(3)
                with cols[0]:
                    if st.button("背景を修正", key="revise_background"):
                        with st.spinner("背景を修正中..."):
                            try:
                                revised_headlines = headline_generator.revise_headline(
                                    headlines,
                                    f"背景について: {background_request}"
                                )
                                st.session_state.headlines.add_revision(
                                    background_request,
                                    revised_headlines
                                )
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"エラーが発生しました: {str(e)}")
                
                with cols[1]:
                    if st.button("課題を修正", key="revise_problem"):
                        with st.spinner("課題を修正中..."):
                            try:
                                revised_headlines = headline_generator.revise_headline(
                                    headlines,
                                    f"課題について: {problem_request}"
                                )
                                st.session_state.headlines.add_revision(
                                    problem_request,
                                    revised_headlines
                                )
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"エラーが発生しました: {str(e)}")
                
                with cols[2]:
                    if st.button("解決策を修正", key="revise_solution"):
                        with st.spinner("解決策を修正中..."):
                            try:
                                revised_headlines = headline_generator.revise_headline(
                                    headlines,
                                    f"解決策について: {solution_request}"
                                )
                                st.session_state.headlines.add_revision(
                                    solution_request,
                                    revised_headlines
                                )
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"エラーが発生しました: {str(e)}")

                # 本文生成セクション
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
                            body = body_generator.generate_body(
                                st.session_state.selected_title,
                                st.session_state.headlines.current,
                                st.session_state.body_prompt
                            )
                            st.session_state.generated_body = BodyRevision(body)
                            st.success("本文を生成しました！")
                        except Exception as e:
                            st.error(f"エラーが発生しました: {str(e)}")

                # 本文表示・修正セクション
                if st.session_state.generated_body:
                    st.subheader("生成された本文")
                    
                    # 本文を段落ごとに分割して表示・修正
                    paragraphs = st.session_state.generated_body.current.split('\n\n')
                    for i, paragraph in enumerate(paragraphs):
                        if paragraph.strip():
                            with st.expander(f"段落 {i+1}", expanded=True):
                                st.write(paragraph)
                                
                                revision_request = st.text_area(
                                    "この段落の修正要望",
                                    key=f"paragraph_revision_{i}",
                                    help="例: より具体的な例を追加、表現を分かりやすく"
                                )
                                
                                if st.button(f"この段落を修正", key=f"revise_paragraph_{i}"):
                                    with st.spinner("段落を修正中..."):
                                        try:
                                            # 段落修正用のプロンプトを作成
                                            specific_request = f"""
以下の段落を修正してください:
{paragraph}

修正要望: {revision_request}
"""
                                            revised_paragraph = body_generator.revise_body(
                                                st.session_state.selected_title,
                                                st.session_state.headlines.current,
                                                paragraph,
                                                specific_request
                                            )
                                            
                                            # 修正した段落を元の本文に反映
                                            paragraphs[i] = revised_paragraph
                                            complete_revised_body = '\n\n'.join(paragraphs)
                                            
                                            # 修正履歴の保存
                                            st.session_state.generated_body.add_revision(
                                                f"段落 {i+1}: {revision_request}",
                                                complete_revised_body
                                            )
                                            
                                            st.success("段落を修正しました！")
                                            st.experimental_rerun()
                                        except Exception as e:
                                            st.error(f"エラーが発生しました: {str(e)}")
                    
                    # 本文全体の修正セクション
                    with st.expander("本文全体の修正", expanded=False):
                        full_revision_request = st.text_area(
                            "本文全体の修正要望",
                            key="full_body_revision",
                            help="例: 全体的な流れを改善、より説得力のある内容に"
                        )
                        
                        if st.button("本文全体を修正"):
                            with st.spinner("本文を修正中..."):
                                try:
                                    revised_body = body_generator.revise_body(
                                        st.session_state.selected_title,
                                        st.session_state.headlines.current,
                                        st.session_state.generated_body.current,
                                        full_revision_request
                                    )
                                    
                                    st.session_state.generated_body.add_revision(
                                        full_revision_request,
                                        revised_body
                                    )
                                    
                                    st.success("本文を修正しました！")
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"エラーが発生しました: {str(e)}")
                    
                    # 修正履歴の表示
                    with st.expander("修正履歴", expanded=False):
                        if st.session_state.generated_body.revision_history:
                            for i, revision in enumerate(st.session_state.generated_body.revision_history, 1):
                                st.write(f"#### 修正 {i}")
                                st.write(f"**修正要望:** {revision['request']}")
                                st.write("**修正前:**")
                                st.write(revision['original'][:200] + "..." if len(revision['original']) > 200 else revision['original'])
                                st.write("**修正後:**")
                                st.write(revision['revised'][:200] + "..." if len(revision['revised']) > 200 else revision['revised'])
                                st.write("---")
                        else:
                            st.info("まだ修正履歴はありません。")

if __name__ == "__main__":
    main()
