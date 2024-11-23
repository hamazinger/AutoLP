import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

import streamlit as st

# Streamlitのページ設定を最初に記述
st.set_page_config(
    page_title="セミナータイトルジェネレーター",
    layout="wide"
)

from google.cloud import bigquery
from google.oauth2 import service_account
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

# OpenAIライブラリのバージョンを確認
st.write(f"OpenAIライブラリのバージョン: {openai.__version__}")

@dataclass
class WebContent:
    title: str
    description: str
    main_content: str
    error: Optional[str] = None

@dataclass
class TitleAnalysis:
    """タイトル分析の結果を格納するデータクラス"""
    predicted_speed: float
    grade: str
    attractive_words: List[str]
    has_specific_problem: bool
    has_exclamation: bool
    title_length: int
    category_score: float
    reasoning: Dict[str, str]

@dataclass
class TitleEvaluation:
    speed: float
    grade: str
    timestamp: str = datetime.now().isoformat()

@dataclass
class GeneratedTitle:
    main_title: str
    sub_title: str
    evaluation: TitleEvaluation

class URLContentExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        
    def extract_with_trafilatura(self, url: str) -> Optional[WebContent]:
        """Trafilaturaを使用してコンテンツを抽出（高精度・推奨）"""
        try:
            downloaded = fetch_url(url)
            if downloaded is None:
                return WebContent(
                    title="",
                    description="",
                    main_content="",
                    error="URLからのコンテンツ取得に失敗しました"
                )
            
            content = extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                timeout=0
            )
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
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.url_extractor = URLContentExtractor()
        self.fixed_prompt_part = """
以下の文脈に基づいて、セミナータイトルを3つ生成してください：

コンテキスト：
{context}

{additional_context}

"""
        # ユーザーが編集できるプロンプト部分
        self.user_editable_prompt = """
以下の条件を満たすタイトルを生成してください：
1. 集客効果の高いキーワード（DX、自動化、セキュリティなど）を適切に含める
2. 具体的な課題や解決方法を明示する
3. タイトルは40文字以内で簡潔にする
4. 感嘆符（！）は使用しない
5. セミナーの価値提案が明確である
6. 製品の特徴や強みを活かしたタイトルにする
"""
        # ユーザーが編集できないプロンプト部分
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください。JSONは有効な形式でなければなりません。
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

    def generate_titles(self, context: str, prompt_template: str = None, product_url: str = None) -> List[Dict[str, str]]:
        """指定されたコンテキストに基づいてタイトルを生成"""
        additional_context = ""
        if product_url:
            content = self.url_extractor.extract_with_trafilatura(product_url)
            if content and not content.error:
                additional_context = f"""
製品タイトル: {content.title}
製品説明: {content.description}
製品詳細: {content.main_content[:1000]}
"""
            else:
                st.warning(f"製品情報の取得に失敗しました: {content.error if content else '不明なエラー'}")
        
        # プロンプトの作成
        prompt = self.fixed_prompt_part.format(
            context=context,
            additional_context=additional_context
        ) + (prompt_template or self.user_editable_prompt) + self.fixed_output_instructions
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # 温度を0に設定
            )
            
            result_text = response.choices[0].message['content'].strip()
            
            # JSON部分を抽出して整形
            try:
                # 最初に直接JSONとしてパースを試みる
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # 失敗した場合、JSONっぽい部分を抽出して再試行
                start_index = result_text.find('{')
                end_index = result_text.rfind('}') + 1
                if start_index != -1 and end_index > start_index:
                    json_text = result_text[start_index:end_index]
                    result = json.loads(json_text)
                else:
                    raise ValueError("タイトルを抽出できませんでした")
    
            # 結果の検証
            if not isinstance(result, dict) or "titles" not in result:
                raise ValueError("不正な応答形式です")
            
            titles = result["titles"]
            if not isinstance(titles, list) or not titles:
                raise ValueError("タイトルが見つかりません")
            
            return titles[:3]  # 最大3つまで
                
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}\nAIからの応答:\n{result_text}")
            return []

class SeminarTitleEvaluator:
    def __init__(self, seminar_data: pd.DataFrame):
        """
        Parameters:
        seminar_data: pd.DataFrame - セミナーデータ
        """
        self.df = seminar_data
        self._initialize_analytics()

    def _initialize_analytics(self):
        """分析に必要なデータを初期化"""
        # 高集客セミナー（2.5以上）からキーワードを抽出
        high_performing = self.df[self.df['Acquisition_Speed'] >= 2.5]
        
        # 効果的なキーワードの抽出
        self.attractive_words = self._extract_effective_keywords(high_performing)
        
        # カテゴリごとの平均集客速度を計算
        self.category_speeds = self.df.groupby('Major_Category')['Acquisition_Speed'].mean()
        
        # 問題提起を示す表現のリスト
        self.problem_indicators = [
            '課題', '問題', 'による', 'ための', '向上', '改善', '解決', '対策',
            'どうする', 'なぜ', 'どう', '方法', '実現', 'ポイント', '実践',
            'ベストプラクティス', 'ノウハウ', '事例', '成功'
        ]

    def _extract_effective_keywords(self, high_performing_df) -> List[str]:
        """効果的なキーワードを抽出"""
        words = []
        for title in high_performing_df['Seminar_Title']:
            if isinstance(title, str):
                # 基本的な前処理
                clean_title = (title.replace('〜', ' ')
                                  .replace('、', ' ')
                                  .replace('【', ' ')
                                  .replace('】', ' ')
                                  .replace('「', ' ')
                                  .replace('」', ' '))
                # キーワードの抽出
                title_words = [w for w in clean_title.split() 
                             if len(w) > 1 and not w.isdigit()]
                words.extend(title_words)
        
        # 出現頻度の高いキーワードを返す
        word_counts = pd.Series(words).value_counts()
        return list(word_counts[word_counts >= 2].index)

    def evaluate_title(self, title: str, category: str = None) -> TitleAnalysis:
        """タイトルを評価して結果を返す"""
        # 基本スコアの計算
        base_score = self._calculate_base_score(title)
        
        # カテゴリによる調整
        category_score = 0.0
        if category and category in self.category_speeds:
            category_avg = self.category_speeds[category]
            category_score = 0.3 if category_avg > 2.5 else (
                0.2 if category_avg > 2.0 else 0.1
            )
        
        # 最終スコアの計算（1.0-3.0の範囲に正規化）
        final_score = min(max(base_score + category_score, 1.0), 3.0)
        
        # マッチした効果的なキーワード
        matching_words = [word for word in self.attractive_words if word in title]
        
        # 問題提起の有無
        has_problem = any(indicator in title for indicator in self.problem_indicators)
        
        # 評価理由の作成
        reasoning = {
            "keywords": f"効果的なキーワード: {', '.join(matching_words) if matching_words else '該当なし'}",
            "title_length": f"タイトルの長さ: {len(title)}文字 （{'適切' if len(title) <= 40 else '長い'}）",
            "problem_indication": f"問題提起: {'あり' if has_problem else 'なし'}",
            "exclamation": f"感嘆符: {'あり（減点）' if '!' in title or '！' in title else 'なし'}",
            "category": f"カテゴリ評価: {category if category else '未指定'} (スコア: {category_score:.1f})",
            "predicted_speed": f"予測される集客速度: {final_score:.1f}"
        }
        
        # グレードの決定
        grade = 'A' if final_score >= 2.5 else 'B' if final_score >= 1.8 else 'C'
        
        return TitleAnalysis(
            predicted_speed=final_score,
            grade=grade,
            attractive_words=matching_words,
            has_specific_problem=has_problem,
            has_exclamation='!' in title or '！' in title,
            title_length=len(title),
            category_score=category_score,
            reasoning=reasoning
        )

    def _calculate_base_score(self, title: str) -> float:
        """基本スコアを計算"""
        base_score = 1.0
        
        # 1. 効果的なキーワードのスコア
        matching_words = [word for word in self.attractive_words if word in title]
        keyword_score = len(matching_words) * 0.4
        base_score += min(keyword_score, 1.2)  # 最大1.2点
        
        # 2. タイトルの長さによる調整
        title_length = len(title)
        if title_length <= 20:
            base_score += 0.3
        elif title_length <= 40:
            base_score += 0.1
        elif title_length > 60:
            base_score -= 0.2
        
        # 3. 問題提起の有無
        if any(indicator in title for indicator in self.problem_indicators):
            base_score += 0.4
        
        # 4. 感嘆符のペナルティ
        if '!' in title or '！' in title:
            base_score -= 0.3
            
        return base_score

class HeadlineGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.fixed_prompt_part = """
以下のセミナータイトルに基づいて、背景・課題・解決策の3つの見出しを生成してください：
「{title}」

"""
        self.user_editable_prompt = """
以下の条件を満たす見出しを生成してください：
- 各見出しは簡潔でわかりやすくする
- セミナーの内容を的確に反映する
"""
        self.fixed_output_instructions = """
以下の形式でJSONを出力してください。余分なテキストは含めず、JSONオブジェクトのみを出力してください。JSONは有効な形式でなければなりません。
{
    "background": "背景の見出し",
    "problem": "課題の見出し",
    "solution": "解決策の見出し"
}
"""
        
    def generate_headlines(self, title: str, prompt_template: str = None) -> Dict[str, str]:
        """タイトルに基づいて見出しを生成"""
        prompt = self.fixed_prompt_part.format(title=title) + (prompt_template or self.user_editable_prompt) + self.fixed_output_instructions
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # 温度を0に設定
            )
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return {}
        
        result_text = response.choices[0].message['content'].strip()
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            start_index = result_text.find('{')
            end_index = result_text.rfind('}') + 1
            json_text = result_text[start_index:end_index]
            result = json.loads(json_text)
        
        return result

class BodyGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.fixed_prompt_part = """
以下のセミナータイトルと見出しに基づいて、本文を生成してください：

タイトル：「{title}」
背景：「{background}」
課題：「{problem}」
解決策：「{solution}」

"""
        self.user_editable_prompt = """
以下の条件を満たす本文を生成してください：
- 各見出しに対して具体的な内容を記述する
- 全体で1000文字以内にまとめる
"""

    def generate_body(self, title: str, headlines: Dict[str, str], prompt_template: str = None) -> str:
        """タイトルと見出しに基づいて本文を生成"""
        prompt = self.fixed_prompt_part.format(
            title=title,
            background=headlines.get("background", ""),
            problem=headlines.get("problem", ""),
            solution=headlines.get("solution", "")
        ) + (prompt_template or self.user_editable_prompt)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # 温度を0に設定
            )
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return ""
        
        result_text = response.choices[0].message['content'].strip()
        return result_text

class InMemoryCache:
    def __init__(self):
        if 'title_cache' not in st.session_state:
            st.session_state.title_cache = {}
    
    def get_evaluation(self, title: str) -> Optional[TitleEvaluation]:
        return st.session_state.title_cache.get(title)
    
    def set_evaluation(self, title: str, evaluation: TitleEvaluation):
        st.session_state.title_cache[title] = evaluation

def init_bigquery_client():
    """BigQueryクライアントの初期化"""
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials)

def load_seminar_data():
    """BigQueryからセミナーデータを読み込む"""
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
        # デバッグ情報の出力
        st.write("データフレームの情報:")
        st.write("カラム一覧:", df.columns.tolist())
        st.write("データ件数:", len(df))
        
        if 'Major_Category' not in df.columns:
            st.error("Major_Categoryカラムが見つかりません")
            st.write("利用可能なカラム:", df.columns.tolist())
            return None
            
        # Major_Categoryの欠損値を確認
        null_categories = df['Major_Category'].isnull().sum()
        st.write("Major_Categoryの欠損値数:", null_categories)
        
        return df
    except Exception as e:
        st.error(f"データの読み込みでエラーが発生しました: {str(e)}")
        return None

def display_evaluation_details(title: str, evaluator: SeminarTitleEvaluator):
    """評価の詳細を表示"""
    analysis = evaluator.evaluate_title(
        title, 
        st.session_state.selected_category
    )
    
    st.write("### 評価詳細")
    
    # 評価理由を表示
    for reason in analysis.reasoning.values():
        st.write(f"- {reason}")
    
    # キーワードのハイライト表示
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

def main():
    init_session_state()
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OpenAI APIキーが設定されていません")
        return
    
    st.title("セミナータイトルジェネレーター")
    
    st.write(f"OpenAIライブラリのバージョン: {openai.__version__}")
    
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
    
    title_generator = TitleGenerator(api_key)
    headline_generator = HeadlineGenerator(api_key)
    body_generator = BodyGenerator(api_key)
    cache = InMemoryCache()
    
    # Step 1: 基本情報入力
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
        category = st.selectbox(
            "カテゴリ",
            options=st.session_state.available_categories
        )
        st.session_state.selected_category = category

    # プロンプト編集機能
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
                    product_url
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
                            grade=analysis.grade
                        )
                        cache.set_evaluation(full_title, evaluation)
                    st.session_state.generated_titles.append(
                        GeneratedTitle(main_title=main_title, sub_title=sub_title, evaluation=evaluation)
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    
    # Step 2: タイトル評価・選択
    if st.session_state.generated_titles:
        st.header("Step 2: タイトル評価・選択")
        
        # 生成されたタイトルの表示
        st.subheader("生成タイトル")
        for i, gen_title in enumerate(st.session_state.generated_titles):
            cols = st.columns([0.5, 2, 2, 1, 1])
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
                                grade=analysis.grade
                            )
                            cache.set_evaluation(full_title, evaluation)
                        st.session_state.generated_titles.append(
                            GeneratedTitle(main_title=manual_main_title, sub_title=manual_sub_title, evaluation=evaluation)
                        )
                        
                        # 評価詳細の表示
                        display_evaluation_details(full_title, st.session_state.evaluator)
                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")
    
    # Step 3: 見出し生成
    if st.session_state.generated_titles:
        st.header("Step 3: 見出し生成")
        
        # 利用可能なタイトルを収集
        available_titles = []
        for gen_title in st.session_state.generated_titles:
            full_title = f"{gen_title.main_title} - {gen_title.sub_title}"
            available_titles.append(full_title)
        
        st.session_state.selected_title_for_headline = st.selectbox(
            "見出しを生成するタイトルを選択してください",
            options=available_titles
        )
        
        # 見出し生成プロンプト編集欄を移動
        with st.expander("見出し生成プロンプトの編集", expanded=False):
            st.session_state.headline_prompt = st.text_area(
                "プロンプトテンプレート",
                value=st.session_state.headline_prompt,
                height=400
            )
        
        if st.button("見出しを生成", key="generate_headlines"):
            with st.spinner("見出しを生成中..."):
                try:
                    st.session_state.headlines = headline_generator.generate_headlines(
                        st.session_state.selected_title_for_headline,
                        st.session_state.headline_prompt
                    )
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
        
        if st.session_state.headlines:
            st.subheader("選択されたタイトル")
            selected_title_eval = next(
                (t.evaluation for t in st.session_state.generated_titles 
                 if f"{t.main_title} - {t.sub_title}" == st.session_state.selected_title_for_headline), 
                None
            )
            
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.write(st.session_state.selected_title_for_headline)
            if selected_title_eval:
                with cols[1]:
                    st.metric("集客速度", f"{selected_title_eval.speed:.1f}")
                with cols[2]:
                    grade_colors = {"A": "green", "B": "orange", "C": "red"}
                    grade_color = grade_colors.get(selected_title_eval.grade, "gray")
                    st.markdown(
                        f'<p style="color: {grade_color}; font-weight: bold; text-align: center;">評価: {selected_title_eval.grade}</p>',
                        unsafe_allow_html=True
                    )
            
            # 見出しの表示
            st.subheader("生成された見出し")
            cols = st.columns(3)
            with cols[0]:
                st.markdown("### 背景")
                st.write(st.session_state.headlines.get("background", ""))
            with cols[1]:
                st.markdown("### 課題")
                st.write(st.session_state.headlines.get("problem", ""))
            with cols[2]:
                st.markdown("### 解決策")
                st.write(st.session_state.headlines.get("solution", ""))
            
            # Step 4: 本文生成
            st.header("Step 4: 本文生成")
            
            # 本文生成プロンプト編集欄
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
                            st.session_state.headlines,
                            st.session_state.body_prompt
                        )
                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")
            
            if st.session_state.generated_body:
                st.subheader("生成された本文")
                st.write(st.session_state.generated_body)

if __name__ == "__main__":
    main()
