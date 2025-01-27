import streamlit as st
from typing import Optional
from ...models.data_classes import GeneratedTitle, TitleEvaluation
from ...services.title_generator import TitleGenerator
from ...utils.cache import InMemoryCache
from ..components.evaluation_display import display_evaluation_details, display_title_row
from ..components.prompt_editor import display_prompt_editor

def render_title_generation_page(title_generator: TitleGenerator, cache: InMemoryCache):
    st.header("Step 1: 基本情報入力")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        product_url = st.text_input("製品URL")
        _handle_product_url(product_url, title_generator)
    with col2:
        pain_points = st.text_area("ペインポイント")
    with col3:
        category = st.selectbox(
            "カテゴリ",
            options=st.session_state.available_categories
        )
        st.session_state.selected_category = category

    file_content = _handle_file_upload()

    st.session_state.title_prompt = display_prompt_editor(
        "タイトル生成",
        st.session_state.title_prompt,
        "title_prompt_editor"
    )

    if st.button("タイトルを生成", key="generate_titles"):
        _generate_titles(
            title_generator,
            cache,
            pain_points,
            category,
            product_url,
            file_content
        )

    if st.session_state.generated_titles:
        _display_generated_titles(cache)

def _generate_titles(title_generator: TitleGenerator, cache: InMemoryCache,
                    pain_points: str, category: str, product_url: str,
                    file_content: Optional[str]):
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
                        original_main_title=main_title,
                        original_sub_title=sub_title
                    )
                )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

def _display_generated_titles(cache: InMemoryCache):
    st.header("Step 2: タイトル評価・選択")
    st.subheader("生成タイトル")

    for i, gen_title in enumerate(st.session_state.generated_titles):
        display_title_row(
            gen_title,
            i,
            {
                "on_select": lambda title: setattr(st.session_state, "selected_title", title),
                "on_refine": lambda idx, prompt: _handle_title_refinement(idx, prompt)
            }
        )

def _handle_title_refinement(index: int, prompt: str):
    if not prompt:
        return

    gen_title = st.session_state.generated_titles[index]
    with st.spinner("タイトル修正中..."):
        refined_title = st.session_state.title_generator.refine_title(
            gen_title.main_title,
            gen_title.sub_title,
            prompt
        )
        if refined_title:
            full_refined_title = f"{refined_title.main_title} - {refined_title.sub_title}"
            analysis = st.session_state.evaluator.evaluate_title(
                full_refined_title,
                st.session_state.selected_category
            )
            evaluation = TitleEvaluation(
                speed=analysis.predicted_speed,
                grade=analysis.grade,
                comment=analysis.evaluation_comment
            )
            st.session_state.generated_titles[index] = GeneratedTitle(
                main_title=refined_title.main_title,
                sub_title=refined_title.sub_title,
                evaluation=evaluation,
                original_main_title=gen_title.original_main_title,
                original_sub_title=gen_title.original_sub_title
            )
            st.rerun()