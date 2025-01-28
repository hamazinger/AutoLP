import streamlit as st
from typing import Optional
from ...models import GeneratedTitle, TitleEvaluation
from ...services import TitleGenerator, SeminarTitleEvaluator
from ..state import init_session_state

def display_title_generation_form(title_generator: TitleGenerator):
    st.header("Step 1: 基本情報入力")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        product_url = st.text_input("製品URL")

    with col2:
        pain_points = st.text_area("ペインポイント")

    with col3:
        category = st.selectbox(
            "カテゴリ",
            options=st.session_state.available_categories
        )
        st.session_state.selected_category = category

    uploaded_file = st.file_uploader("ファイルをアップロード", type=['txt', 'pdf', 'docx'])
    file_content = None
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
            st.error(f"ファイルの読み込みでエラーが発生しました: {e}")

    with st.expander("タイトル生成プロンプトの編集", expanded=False):
        st.session_state.title_prompt = st.text_area(
            "プロンプトテンプレート",
            value=st.session_state.title_prompt,
            height=400
        )

    if st.button("タイトルを生成"):
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
                    analysis = st.session_state.evaluator.evaluate_title(full_title, category)
                    evaluation = TitleEvaluation(
                        speed=analysis.predicted_speed,
                        grade=analysis.grade,
                        comment=analysis.evaluation_comment
                    )
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

def display_generated_titles(title_generator: TitleGenerator):
    if st.session_state.generated_titles:
        st.header("Step 2: タイトル評価・選択")
        st.subheader("生成タイトル")

        for i, gen_title in enumerate(st.session_state.generated_titles):
            cols = st.columns([0.5, 2, 2, 1, 1, 2, 1])
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

            with cols[5]:
                st.write(f"**評価:** {gen_title.evaluation.comment}")

            with cols[6]:
                refine_prompt = st.text_area(
                    "修正依頼",
                    key=f"refine_prompt_{i}",
                    height=70,
                    label_visibility="collapsed",
                    placeholder="例：もっと具体的に"
                )
                if st.button("修正", key=f"refine_button_{i}"):
                    with st.spinner("タイトル修正中..."):
                        try:
                            refined_title = title_generator.refine_title(
                                gen_title.main_title,
                                gen_title.sub_title,
                                refine_prompt
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
                                st.session_state.generated_titles[i] = GeneratedTitle(
                                    main_title=refined_title.main_title,
                                    sub_title=refined_title.sub_title,
                                    evaluation=evaluation,
                                    original_main_title=gen_title.original_main_title,
                                    original_sub_title=gen_title.original_sub_title
                                )
                                st.rerun()
                        except Exception as e:
                            st.error(f"タイトル修正でエラーが発生しました: {e}")

def display_title_evaluation(evaluator: SeminarTitleEvaluator):
    st.write("### 評価詳細")

    manual_main_title = st.text_input("メインタイトル", key="manual_main_title")
    manual_sub_title = st.text_input("サブタイトル", key="manual_sub_title")

    if manual_main_title and manual_sub_title:
        full_title = f"{manual_main_title} - {manual_sub_title}"
        analysis = evaluator.evaluate_title(
            full_title,
            st.session_state.selected_category
        )

        st.info(f"**評価コメント:** {analysis.evaluation_comment}")

        for reason in analysis.reasoning.values():
            st.write(f"- {reason}")

        if analysis.attractive_words:
            st.write("### タイトル中の効果的なキーワード")
            highlighted_title = full_title
            for word in analysis.attractive_words:
                highlighted_title = highlighted_title.replace(
                    word,
                    f'<span style="background-color: #FFEB3B">{word}</span>'
                )
            st.markdown(f'<p>{highlighted_title}</p>', unsafe_allow_html=True)