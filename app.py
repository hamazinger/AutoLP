import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

# 前のインポートとクラス定義は省略...

class ContentReviser:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model,
            openai_api_key=api_key
        )

    def revise_content(self, original_content: str, revision_request: str, content_type: str) -> str:
        prompt_templates = {
            "title": """
            あなたはセミナータイトルの編集者です。
            以下のセミナータイトルを、ユーザーの要望に基づいて修正してください。
            
            現在のタイトル：
            {original_content}
            
            修正要望：
            {revision_request}
            
            以下の制約条件を守ってください：
            - メインタイトルとサブタイトルの形式を維持
            - 各タイトルは40文字以内
            - 感嘆符は使用しない
            """,
            "headline": """
            あなたはセミナー見出しの編集者です。
            以下の見出しを、ユーザーの要望に基づいて修正してください。
            
            現在の見出し：
            {original_content}
            
            修正要望：
            {revision_request}
            
            見出しの役割を維持したまま修正してください。
            """,
            "body": """
            あなたはセミナー本文の編集者です。
            以下の本文を、ユーザーの要望に基づいて修正してください。
            
            現在の本文：
            {original_content}
            
            修正要望：
            {revision_request}
            
            以下の制約条件を守ってください：
            - 見出しの構造を維持
            - 各セクション300文字以上
            - 全体で1000文字以内
            - 箇条書きを使用しない
            """
        }

        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=prompt_templates[content_type],
                input_variables=["original_content", "revision_request"]
            )
        )

        try:
            return chain.run({
                "original_content": original_content,
                "revision_request": revision_request
            })
        except Exception as e:
            st.error(f"コンテンツの修正中にエラーが発生しました: {str(e)}")
            return original_content

class EnhancedTitleGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=api_key
        )
        self.data_processor = SeminarDataProcessor()

    def generate_titles(self, context: str, category: str):
        similar_seminars = self.data_processor.find_similar_seminars(
            f"{context} {category}",
            k=3
        )

        template = """
        以下の情報を基に、セミナータイトルを生成してください：

        コンテキスト：
        {context}

        カテゴリ：
        {category}

        過去の類似セミナー：
        {similar_seminars}

        以下の特徴を持つ成功しているセミナータイトルの特徴を考慮してください：
        1. 平均集客速度が{avg_speed}以上
        2. 平均参加者数が{avg_participants}人以上
        3. 平均レスポンス率が{avg_response_rate}%以上

        生成するタイトルの要件：
        - メインタイトルとサブタイトルに分ける
        - メインタイトルでは問題点や課題を投げかける
        - サブタイトルでは解決策やベネフィットを表現する
        - 各タイトルは40文字以内
        - 感嘆符は使用しない

        以下の形式でJSONを出力してください：
        {
            "titles": [
                {
                    "main_title": "メインタイトル1",
                    "sub_title": "サブタイトル1"
                },
                {
                    "main_title": "メインタイトル2",
                    "sub_title": "サブタイトル2"
                },
                {
                    "main_title": "メインタイトル3",
                    "sub_title": "サブタイトル3"
                }
            ]
        }
        """

        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=template,
                input_variables=[
                    "context", "category", "similar_seminars",
                    "avg_speed", "avg_participants", "avg_response_rate"
                ]
            )
        )

        result = chain.run({
            "context": context,
            "category": category,
            "similar_seminars": "\n".join([doc.page_content for doc in similar_seminars]),
            "avg_speed": 2.5,
            "avg_participants": 35,
            "avg_response_rate": 50
        })

        try:
            return json.loads(result)["titles"]
        except Exception as e:
            st.error(f"タイトルの生成中にエラーが発生しました: {str(e)}")
            return []