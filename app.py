#前半部分は省略...

            st.error(f"タイトルの生成中にエラーが発生しました: {str(e)}")
            return []

def display_content_with_revision(content: str, content_type: str, reviser: ContentReviser):
    """修正機能付きでコンテンツを表示するコンポーネント"""
    st.markdown(f"### 生成された{content_type}")
    st.write(content)
    
    with st.expander("内容を修正する", expanded=False):
        revision_request = st.text_area(
            "修正したい内容を自然な言葉で入力してください",
            key=f"revision_{content_type}",
            help="例：「もう少し具体的にしてください」「簡潔にまとめてください」など"
        )
        
        if st.button("修正する", key=f"revise_{content_type}"):
            with st.spinner("修正中..."):
                revised_content = reviser.revise_content(content, revision_request, content_type)
                st.session_state[f"{content_type}_content"] = revised_content
                st.success("修正が完了しました")
                st.write("修正後の内容:")
                st.write(revised_content)

def get_relevant_industry_data(df):
    """業界関連データの取得"""
    return df.to_dict('records')[:10]  # サンプルとして最初の10件を返す

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
    if 'product_analysis' not in st.session_state:
        st.session_state.product_analysis = None
    if 'pain_point_analysis' not in st.session_state:
        st.session_state.pain_point_analysis = None

def main():
    init_session_state()
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OpenAI APIキーが設定されていません")
        return
    
    st.title("セミナータイトルジェネレーター")
    
    content_analyzer = EnhancedContentAnalyzer(api_key)
    content_reviser = ContentReviser(api_key)
    
    if st.session_state.seminar_data is None:
        with st.spinner("セミナーデータを読み込んでいます..."):
            df = load_seminar_data()
            if df is not None:
                try:
                    categories = df['Major_Category'].dropna().unique().tolist()
                    st.session_state.available_categories = sorted(categories)
                    st.session_state.seminar_data = df
                    
                    # LangChainによるデータ処理の初期化
                    data_processor = SeminarDataProcessor()
                    data_processor.process_historical_data(df)
                    st.session_state.data_processor = data_processor
                    
                    st.success("データを正常に読み込みました！")
                except Exception as e:
                    st.error(f"データの処理中にエラーが発生しました: {str(e)}")
                    return
    
    st.header("Step 1: 基本情報入力")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        product_url = st.text_input("製品URL")
        if product_url:
            with st.spinner("製品情報を分析中..."):
                product_analysis = content_analyzer.analyze_product_url(product_url)
                if product_analysis:
                    st.success("製品情報の分析が完了しました")
                    with st.expander("分析結果を表示"):
                        st.json(product_analysis)
                    st.session_state.product_analysis = product_analysis
    
    with col2:
        pain_points = st.text_area("ペインポイント")
        category = st.selectbox(
            "カテゴリ",
            options=st.session_state.available_categories
        )
        st.session_state.selected_category = category
        
        if pain_points and st.session_state.seminar_data is not None:
            if st.button("ペインポイントを分析"):
                with st.spinner("ペインポイントを分析中..."):
                    industry_data = get_relevant_industry_data(st.session_state.seminar_data)
                    pain_point_analysis = content_analyzer.analyze_pain_points(
                        pain_points,
                        industry_data
                    )
                    st.success("ペインポイントの分析が完了しました")
                    with st.expander("分析結果を表示"):
                        st.json(pain_point_analysis)
                    st.session_state.pain_point_analysis = pain_point_analysis

    # タイトル生成と修正UI
    title_generator = EnhancedTitleGenerator(api_key)
    
    if st.button("タイトルを生成"):
        context = f"""
        製品情報: {st.session_state.product_analysis if st.session_state.product_analysis else ''}
        ペインポイント: {pain_points}
        カテゴリ: {category}
        """
        titles = title_generator.generate_titles(context, category)
        st.session_state.generated_titles = titles
    
    if st.session_state.generated_titles:
        st.header("Step 2: 生成されたタイトル")
        for i, title in enumerate(st.session_state.generated_titles):
            full_title = f"{title['main_title']} - {title['sub_title']}"
            display_content_with_revision(full_title, f"title_{i}", content_reviser)

if __name__ == "__main__":
    main()