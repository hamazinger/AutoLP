import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

import streamlit as st
from openai import OpenAI

from .utils.bigquery_client import load_seminar_data
from .utils.cache import InMemoryCache
from .services.title_generator import TitleGenerator
from .services.title_evaluator import SeminarTitleEvaluator
from .services.headline_generator import HeadlineGenerator
from .services.body_generator import BodyGenerator
from .ui.pages import (
    render_title_generation_page,
    render_headline_generation_page,
    render_body_generation_page
)

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
    if 'title_generator' not in st.session_state:
        st.session_state.title_generator = None

def main():
    st.set_page_config(
        page_title="セミナータイトルジェネレーター",
        layout="wide"
    )

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
    if st.session_state.title_generator is None:
        st.session_state.title_generator = TitleGenerator(api_key, model=model_name)
    
    headline_generator = HeadlineGenerator(api_key, model=model_name)
    body_generator = BodyGenerator(api_key, model=model_name)
    cache = InMemoryCache()

    # タイトル生成ページのレンダリング
    render_title_generation_page(st.session_state.title_generator, cache)

    # 見出し生成ページのレンダリング
    render_headline_generation_page(headline_generator)

    # 本文生成ページのレンダリング
    render_body_generation_page(body_generator)

if __name__ == "__main__":
    main()