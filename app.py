import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

# 前のインポートと基本クラス定義は省略...

class EnhancedContentAnalyzer:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def analyze_product_url(self, url: str):
        try:
            loader = WebBaseLoader(url)
            document = loader.load()
            
            texts = self.text_splitter.split_documents(document)
            
            product_analysis_template = """
            以下の製品情報から重要な要素を抽出・分析してください：

            {text}

            以下の形式でJSONとして出力してください：
            {
                "product_name": "製品名",
                "key_features": ["主要な機能1", "主要な機能2"...],
                "target_users": ["対象ユーザー1", "対象ユーザー2"...],
                "pain_points": ["解決する課題1", "解決する課題2"...],
                "benefits": ["提供価値1", "提供価値2"...],
                "technical_details": ["技術的特徴1", "技術的特徴2"...]
            }
            """
            
            prompt = PromptTemplate(
                template=product_analysis_template,
                input_variables=["text"]
            )
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=prompt,
                combine_prompt=prompt
            )
            
            return chain.run(texts)
            
        except Exception as e:
            st.error(f"製品情報の分析中にエラーが発生しました: {str(e)}")
            return None

    def analyze_pain_points(self, pain_points: str, industry_data: list):
        pain_point_template = """
        以下のペインポイントを分析し、業界コンテキストと関連付けてください：

        ペインポイント：
        {pain_points}

        業界データ：
        {industry_data}

        以下の観点で分析してください：
        1. 課題の重要度
        2. 業界での一般性
        3. 解決の緊急性
        4. 潜在的な影響範囲
        5. 類似事例との関連性

        出力形式：
        {
            "priority_level": "重要度（高/中/低）",
            "industry_relevance": "業界での一般性の説明",
            "urgency": "解決の緊急性の説明",
            "impact_range": "影響範囲の分析",
            "similar_cases": ["関連する類似事例1", "関連する類似事例2"...],
            "recommended_approaches": ["推奨アプローチ1", "推奨アプローチ2"...]
        }
        """

        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=pain_point_template,
                input_variables=["pain_points", "industry_data"]
            )
        )

        return chain.run({
            "pain_points": pain_points,
            "industry_data": str(industry_data)
        })

class SeminarDataProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    def process_historical_data(self, df):
        seminar_texts = []
        for _, row in df.iterrows():
            text = f"""
            タイトル: {row['Seminar_Title']}
            カテゴリ: {row['Major_Category']}
            集客速度: {row['Acquisition_Speed']}
            参加者数: {row['Total_Participants']}
            レスポンス率: {row['Action_Response_Rate']}
            """
            seminar_texts.append(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = text_splitter.create_documents(seminar_texts)

        self.vector_store = Chroma.from_documents(
            documents, 
            self.embeddings
        )

    def find_similar_seminars(self, query_text, k=5):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query_text, k=k)