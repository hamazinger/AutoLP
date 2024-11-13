import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# BigQueryクライアントの初期化
def init_bigquery_client():
    """BigQueryクライアントの初期化"""
    # サービスアカウントキーの情報をst.secretsから取得
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials)

def load_seminar_data():
    """BigQueryからセミナーデータを読み込む"""
    client = init_bigquery_client()
    
    query = """
    SELECT 
        Seminar_Title,
        Acquisition_Speed,
        Major_Category,
        Category,
        Total_Participants,
        Action_Response_Count,
        Action_Response_Rate,
        User_Company_Percentage,
        Non_User_Company_Percentage
    FROM `mythical-envoy-386309.majisemi.majisemi_seminar_usukiapi`
    WHERE Seminar_Title IS NOT NULL
    AND Acquisition_Speed IS NOT NULL
    """
    
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"データの読み込みでエラーが発生しました: {str(e)}")
        return None

# データクラスの定義
@dataclass
class TitleEvaluation:
    speed: float
    grade: str
    timestamp: str = datetime.now().isoformat()

@dataclass
class GeneratedTitle:
    title: str
    evaluation: TitleEvaluation

# インメモリキャッシュ
class InMemoryCache:
    def __init__(self):
        if 'title_cache' not in st.session_state:
            st.session_state.title_cache = {}
    
    def get_evaluation(self, title: str) -> Optional[TitleEvaluation]:
        return st.session_state.title_cache.get(title)
    
    def set_evaluation(self, title: str, evaluation: TitleEvaluation):
        st.session_state.title_cache[title] = evaluation

class TitleGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_titles(self, context: str) -> List[str]:
        """指定されたコンテキストに基づいてタイトルを生成"""
        prompt = f"""
        以下の文脈に基づいて、セミナータイトルを3つ生成してください：
        
        コンテキスト：
        {context}
        
        以下の条件を満たすタイトルを生成してください：
        1. 集客効果の高いキーワード（DX、自動化、セキュリティなど）を適切に含める
        2. 具体的な課題や解決方法を明示する
        3. タイトルは40文字以内で簡潔にする
        4. 感嘆符（！）は使用しない
        5. セミナーの価値提案が明確である
        
        以下の形式でJSONを出力してください：
        {{
            "titles": [
                "タイトル1",
                "タイトル2",
                "タイトル3"
            ]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["titles"]

class TitleEvaluator:
    def __init__(self, api_key: str, seminar_data: pd.DataFrame = None):
        self.evaluator = SeminarTitleEvaluator(seminar_data) if seminar_data is not None else None
    
    def evaluate_title(self, title: str, category: str = None) -> TitleEvaluation:
        if self.evaluator is None:
            raise ValueError("セミナーデータが読み込まれていません")
        
        analysis = self.evaluator.evaluate_title(title, category)
        return TitleEvaluation(
            speed=analysis.predicted_speed,
            grade=analysis.grade,
            timestamp=datetime.now().isoformat()
        )

def init_session_state():
    """セッション状態の初期化"""
    if 'generated_titles' not in st.session_state:
        st.session_state.generated_titles = []
    if 'selected_title' not in st.session_state:
        st.session_state.selected_title = None
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

def main():
    st.set_page_config(
        page_title="セミナータイトルジェネレーター",
        layout="wide"
    )
    
    init_session_state()
    
    # APIキーの設定
    api_key = st.secrets["OPENAI_API_KEY"]
    
    st.title("セミナータイトルジェネレーター")
    
    # BigQueryからデータを読み込む
    if st.session_state.seminar_data is None:
        with st.spinner("セミナーデータを読み込んでいます..."):
            df = load_seminar_data()
            if df is not None:
                st.session_state.seminar_data = df
                st.session_state.evaluator = TitleEvaluator(api_key, df)
                st.success("データを正常に読み込みました！")
            else:
                st.error("データの読み込みに失敗しました。")
                return
    
    # サービスの初期化
    title_generator = TitleGenerator(api_key)
    headline_generator = HeadlineGenerator(api_key)
    cache = InMemoryCache()
    
    # Step 1: 基本情報入力
    st.header("Step 1: 基本情報入力")
    
    # カテゴリ選択肢の動的生成
    available_categories = sorted(st.session_state.seminar_data['Major_Category'].unique())
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        product_link = st.text_input("製品リンク")
    with col2:
        pain_points = st.text_area("ペインポイント")
    with col3:
        category = st.selectbox(
            "カテゴリ",
            options=available_categories
        )
        st.session_state.selected_category = category
    
    # ... (以下のコードは変更なし) ...

if __name__ == "__main__":
    main()
