import streamlit as st
from ...services.headline_generator import HeadlineGenerator
from ...models.data_classes import HeadlineSet
from ..components.prompt_editor import display_prompt_editor

def render_headline_generation_page(headline_generator: HeadlineGenerator):
    if not st.session_state.generated_titles:
        return

    st.header("Step 3: 見出し生成")

    available_titles = [
        f"{gen_title.main_title} - {gen_title.sub_title}"
        for gen_title in st.session_state.generated_titles
    ]

    st.session_state.selected_title_for_headline = st.selectbox(
        "見出しを生成するタイトルを選択してください",
        options=available_titles
    )

    st.session_state.headline_prompt = display_prompt_editor(
        "見出し生成",
        st.session_state.headline_prompt,
        "headline_prompt_editor"
    )

    if st.button("見出しを生成", key="generate_headlines"):
        _generate_headlines(headline_generator)

    if st.session_state.manual_headlines:
        _display_headline_editor()

def _generate_headlines(headline_generator: HeadlineGenerator):
    with st.spinner("見出しを生成中..."):
        try:
            headlines = headline_generator.generate_headlines(
                st.session_state.selected_title_for_headline,
                st.session_state.headline_prompt
            )
            st.session_state.headlines = headlines
            st.session_state.manual_headlines = headlines
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

def _display_headline_editor():
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