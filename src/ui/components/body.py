import streamlit as st
from ...services import BodyGenerator

def display_body_generation_form(body_generator: BodyGenerator):
    if st.session_state.manual_headlines:
        st.header("Step 4: 本文生成")

        with st.expander("本文生成プロンプトの編集", expanded=False):
            st.session_state.body_prompt = st.text_area(
                "本文生成プロンプトテンプレート",
                value=st.session_state.body_prompt,
                height=400
            )

        if st.button("本文を生成"):
            with st.spinner("本文を生成中..."):
                try:
                    st.session_state.generated_body = body_generator.generate_body(
                        st.session_state.selected_title_for_headline,
                        st.session_state.manual_headlines,
                        st.session_state.body_prompt
                    )
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

def display_generated_body():
    if st.session_state.generated_body:
        st.subheader("生成された本文")
        st.markdown(st.session_state.generated_body)