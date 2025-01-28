                        st.session_state.generated_titles.append(
                            GeneratedTitle(
                                main_title=manual_main_title,
                                sub_title=manual_sub_title,
                                evaluation=evaluation,
                                original_main_title=manual_main_title,
                                original_sub_title=manual_sub_title
                            )
                        )

                        display_evaluation_details(full_title, st.session_state.evaluator)
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

        # Step 3: 見出し生成
        if st.session_state.generated_titles:
            st.header("Step 3: 見出し生成")

            available_titles = []
            for gen_title in st.session_state.generated_titles:
                full_title = f"{gen_title.main_title} - {gen_title.sub_title}"
                available_titles.append(full_title)

            st.session_state.selected_title_for_headline = st.selectbox(
                "見出しを生成するタイトルを選択してください",
                options=available_titles
            )

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
                            st.session_state.selected_title_for_headline,
                            st.session_state.headline_prompt
                        )
                        st.session_state.headlines = headlines
                        st.session_state.manual_headlines = headlines
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")

            if st.session_state.manual_headlines:
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

                # Step 4: 本文生成
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
                            st.session_state.generated_body = body_generator.generate_body(
                                st.session_state.selected_title_for_headline,
                                st.session_state.manual_headlines,
                                st.session_state.body_prompt
                            )
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")

                if st.session_state.generated_body:
                    st.subheader("生成された本文")
                    st.write(st.session_state.generated_body)

if __name__ == "__main__":
    main()