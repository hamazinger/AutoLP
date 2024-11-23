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
    title: str
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
        self.default_prompt = """
        以下の文脈に基づいて、セミナータイトルを3つ生成してください：

        コンテキスト：
        {context}

        {additional_context}

        以下の条件を満たすタイトルを生成してください：
        1. 集客効果の高いキーワード（DX、自動化、セキュリティなど）を適切に含める
        2. 具体的な課題や解決方法を明示する
        3. タイトルは40文字以内で簡潔にする
        4. 感嘆符（！）は使用しない
        5. セミナーの価値提案が明確である
        6. 製品の特徴や強みを活かしたタイトルにする

        以下の形式でJSONを出力してください：
        {
            "titles": [
                "タイトル1",
                "タイトル2",
                "タイトル3"
            ]
        }
        """
    
    def generate_titles(self, context: str, prompt_template: str = None, product_url: str = None) -> List[str]:
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
        prompt = (prompt_template or self.default_prompt).format(
            context=context,
            additional_context=additional_context
        )
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
        except Exception as e:
            st.error(f"OpenAI APIの呼び出しでエラーが発生しました: {str(e)}")
            return []
        
        result_text = response.choices[0].message['content'].strip()
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            start_index = result_text.find('{')
            end_index = result_text.rfind('}') + 1
            json_text = result_text[start_index:end_index]
            result = json.loads(json_text)
        
        return result["titles"]

class HeadlineGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.default_prompt = """
        以下のセミナータイトルに基づいて、背景・課題・解決策の3つの見出しを生成してください：
        「{title}」

        以下の形式でJSONを出力してください：
        {
            "background": "背景の見出し",
            "problem": "課題の見出し",
            "solution": "解決策の見出し"
        }
        """
    
    def generate_headlines(self, title: str, prompt_template: str = None) -> Dict[str, str]:
        """タイトルに基づいて見出しを生成"""
        prompt = (prompt_template or self.default_prompt).format(title=title)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは優秀なコピーライターです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
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

[... Rest of the existing classes remain unchanged ...]

def init_session_state():
    """セッション状態の初期化"""
    if 'generated_titles' not in st.session_state:
        st.session_state.generated_titles = []
    if 'selected_title' not in st.session_state:
        st.session_state.selected_title = None
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
        st.session_state.title_prompt = TitleGenerator("dummy_key").default_prompt
    if 'headline_prompt' not in st.session_state:
        st.session_state.headline_prompt = HeadlineGenerator("dummy_key").default_prompt

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
                    cached_eval = cache.get_evaluation(title)
                    if cached_eval:
                        evaluation = cached_eval
                    else:
                        analysis = st.session_state.evaluator.evaluate_title(title, category)
                        evaluation = TitleEvaluation(
                            speed=analysis.predicted_speed,
                            grade=analysis.grade
                        )
                        cache.set_evaluation(title, evaluation)
                    st.session_state.generated_titles.append(
                        GeneratedTitle(title=title, evaluation=evaluation)
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    
    
    # Step 3: 見出し生成
    if st.session_state.selected_title:
        st.header("Step 3: 見出し生成")
        
        if st.button("見出しを生成", key="generate_headlines"):
            with st.spinner("見出しを生成中..."):
                try:
                    st.session_state.headlines = headline_generator.generate_headlines(
                        st.session_state.selected_title
                    )
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
        
        if st.session_state.headlines:
            st.subheader("選択されたタイトル")
            selected_title_eval = next(
                (t.evaluation for t in st.session_state.generated_titles 
                 if t.title == st.session_state.selected_title), 
                None
            )
            
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.write(st.session_state.selected_title)
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
                st.write(st.session_state.headlines["background"])
            with cols[1]:
                st.markdown("### 課題")
                st.write(st.session_state.headlines["problem"])
            with cols[2]:
                st.markdown("### 解決策")
                st.write(st.session_state.headlines["solution"])

if __name__ == "__main__":
    main()
