import streamlit as st
from ...services.body_generator import BodyGenerator
from ..components.prompt_editor import display_prompt_editor

def render_body_generation_page(body_generator: BodyGenerator):
    if not st.session_state.manual_headlines:
        return

    st.header("Step 4: 本文生成")

    st.session_state.body_prompt = display_prompt_editor(
        "本文生成",
        st.session_state.body_prompt,
        "body_prompt_editor"
    )

    if st.button("本文を生成", key="generate_body"):
        _generate_body(body_generator)

    if st.session_state.generated_body:
        st.subheader("生成された本文")
        st.write(st.session_state.generated_body)

def _generate_body(body_generator: BodyGenerator):
    with st.spinner("本文を生成中..."):
        try:
            st.session_state.generated_body = body_generator.generate_body(
                st.session_state.selected_title_for_headline,
                st.session_state.manual_headlines,
                st.session_state.body_prompt
            )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")