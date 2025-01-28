import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

import streamlit as st
from ..config import Settings
from ..data import load_seminar_data, InMemoryCache
from ..services import (
    TitleGenerator,
    HeadlineGenerator,
    BodyGenerator,
    SeminarTitleEvaluator
)
from .state import init_session_state
from .components import (
    display_title_generation_form,
    display_generated_titles,
    display_title_evaluation,
    display_headline_generation_form,
    display_generated_headlines,
    display_body_generation_form,
    display_generated_body
)

def run_app():
    # 初期設定
    settings = Settings()
    st.set_page_config(**settings.page_config)
    st.markdown(settings.hide_streamlit_style, unsafe_allow_html=True)
    init_session_state()

    # 各サービスの初期化
    title_generator = TitleGenerator(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MODEL_NAME
    )
    headline_generator = HeadlineGenerator(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MODEL_NAME
    )
    body_generator = BodyGenerator(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MODEL_NAME
    )
    cache = InMemoryCache()

    # データのロード
    if st.session_state.seminar_data is None:
        with st.spinner("セミナーデータを読み込んでいます..."):
            try:
                df = load_seminar_data()
                categories = df['Major_Category'].dropna().unique().tolist()
                st.session_state.available_categories = sorted(categories)
                st.session_state.seminar_data = df
                st.session_state.evaluator = SeminarTitleEvaluator(df)
                st.success("データを正常に読み込みました！")
            except Exception as e:
                st.error(f"データの読み込みに失敗しました: {e}")
                return

    # タイトルジェネレーターのタイトル
    st.title(settings.PROJECT_NAME)

    # Step 1: 基本情報入力
    display_title_generation_form(title_generator)

    # Step 2: タイトル評価・選択
    display_generated_titles(title_generator)
    display_title_evaluation(st.session_state.evaluator)

    # Step 3: 見出し生成
    display_headline_generation_form(headline_generator)
    display_generated_headlines()

    # Step 4: 本文生成
    display_body_generation_form(body_generator)
    display_generated_body()

if __name__ == "__main__":
    run_app()