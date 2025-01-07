import os
os.environ["TRAFILATURA_USE_SIGNAL"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json
import openai
import requests
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract
from PyPDF2 import PdfReader
from docx import Document
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import AnalyzeDocumentChain, LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Streamlitのページ設定
st.set_page_config(
    page_title="セミナータイトルジェネレーター",
    layout="wide"
)

# 右下の開発者プロフィールリンクやフッター非表示用CSS
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# データクラスの定義
@dataclass
class WebContent:
    title: str
    description: str
    main_content: str
    error: Optional[str] = None

@dataclass
class TitleAnalysis:
    predicted_speed: float
    grade: str
    attractive_words: List[str]
    has_specific_problem: bool
    has_exclamation: bool
    title_length: int
    category_score: float
    reasoning: Dict[str, str]
    evaluation_comment: str

@dataclass
class TitleEvaluation:
    speed: float
    grade: str
    comment: str
    timestamp: str = datetime.now().isoformat()

@dataclass
class GeneratedTitle:
    main_title: str
    sub_title: str
    evaluation: TitleEvaluation

@dataclass
class HeadlineSet:
    background: str
    problem: str
    solution: str
    
    def to_dict(self):
        return {
            "background": self.background,
            "problem": self.problem,
            "solution": self.solution
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            background=data.get("background", ""),
            problem=data.get("problem", ""),
            solution=data.get("solution", "")
        )

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
        """製品URLの内容を分析して構造化された情報を抽出"""
        try:
            # WebLoaderを使用してコンテンツを取得
            loader = WebBaseLoader(url)
            document = loader.load()
            
            # コンテンツを分析可能なチャンクに分割
            texts = self.text_splitter.split_documents(document)
            
            # 製品情報抽出用のプロンプト
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
            
            # 分析チェーンの作成
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
        """ペインポイントを分析し、業界コンテキストと関連付け"""
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
        # セミナーデータをテキストに変換
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

        # テキストを分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = text_splitter.create_documents(seminar_texts)

        # ベクターストアの作成
        self.vector_store = Chroma.from_documents(
            documents, 
            self.embeddings
        )

    def find_similar_seminars(self, query_text, k=5):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query_text, k=k)

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
        # 類似セミナーの検索
        similar_seminars = self.data_processor.find_similar_seminars(
            f"{context} {category}",
            k=3
        )

        # プロンプトテンプレート
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
