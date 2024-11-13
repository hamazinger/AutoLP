import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Optional
import openai
from openai import OpenAI
from dataclasses import dataclass

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
        prompt = f"""
        以下の文脈に基づいて、セミナータイトルを3つ生成してください：
        
        コンテキスト：
        {context}
        
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
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def evaluate_title(self, title: str) -> TitleEvaluation:
        prompt = f"""
        以下のセミナータイトルを評価してください：
        「{title}」

        以下の形式でJSONを出力してください：
        {{
            "speed": float,  # 1.0-3.0の範囲で集客速度を評価
            "grade": str     # "A", "B", "C"のいずれか
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        result = json.loads(response.choices[0].message.content)
        return TitleEvaluation(
            speed=result["speed"],
            grade=result["grade"]
        )

class HeadlineGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def generate_headlines(self, title: str) -> Dict[str, str]:
        prompt = f"""
        以下のセミナータイトルに基づいて、背景・課題・解決策の3つの見出しを生成してください：
        「{title}」
        
        以下の形式でJSONを出力してください：
        {{
            "background": str,  # 背景の見出し
            "problem": str,     # 課題の見出し
            "solution": str     # 解決策の見出し
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)

def init_session_state():
    if 'generated_titles' not in st.session_state:
        st.session_state.generated_titles = []
    if 'selected_title' not in st.session_state:
        st.session_state.selected_title = None
    if 'headlines' not in st.session_state:
        st.session_state.headlines = None
    if 'title_cache' not in st.session_state:
        st.session_state.title_cache = {}

def main():
    st.set_page_config(page_title="セミナータイトルジェネレーター", layout="wide")
    
    init_session_state()
    
    # APIキーの設定
    api_key = st.secrets["OPENAI_API_KEY"]
    
    # サービスの初期化
    title_generator = TitleGenerator(api_key)
    title_evaluator = TitleEvaluator(api_key)
    headline_generator = HeadlineGenerator(api_key)
    cache = InMemoryCache()
    
    st.title("セミナータイトルジェネレーター")
    
    # Step 1: 基本情報入力
    st.header("Step 1: 基本情報入力")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        product_link = st.text_input("製品リンク")
    with col2:
        pain_points = st.text_area("ペインポイント")
    
    if st.button("タイトルを生成", key="generate_titles"):
        context = f"製品リンク: {product_link}\nペインポイント: {pain_points}"
        with st.spinner("タイトルを生成中..."):
            try:
                titles = title_generator.generate_titles(context)
                st.session_state.generated_titles = []
                for title in titles:
                    # キャッシュチェック
                    cached_eval = cache.get_evaluation(title)
                    if cached_eval:
                        evaluation = cached_eval
                    else:
                        evaluation = title_evaluator.evaluate_title(title)
                        cache.set_evaluation(title, evaluation)
                    st.session_state.generated_titles.append(
                        GeneratedTitle(title=title, evaluation=evaluation)
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    
    # Step 2: タイトル評価・選択
    if st.session_state.generated_titles:
        st.header("Step 2: タイトル評価・選択")
        
        # 生成されたタイトルの表示
        st.subheader("生成タイトル")
        for i, gen_title in enumerate(st.session_state.generated_titles):
            cols = st.columns([0.5, 3, 1, 1])
            with cols[0]:
                if st.radio(
                    "選択",
                    ["✓"],
                    key=f"radio_{i}",
                    label_visibility="collapsed"
                ):
                    st.session_state.selected_title = gen_title.title
            with cols[1]:
                st.write(gen_title.title)
            with cols[2]:
                st.metric("集客速度", f"{gen_title.evaluation.speed:.1f}")
            with cols[3]:
                grade_colors = {"A": "green", "B": "orange", "C": "red"}
                grade_color = grade_colors.get(gen_title.evaluation.grade, "gray")
                st.markdown(
                    f'<p style="color: {grade_color}; font-weight: bold; text-align: center;">評価: {gen_title.evaluation.grade}</p>',
                    unsafe_allow_html=True
                )
        
        # 手動タイトル評価
        st.subheader("手動タイトル評価")
        col1, col2 = st.columns([4, 1])
        with col1:
            manual_title = st.text_input("評価したいタイトル", key="manual_title")
        with col2:
            if st.button("評価する", key="evaluate_manual") and manual_title:
                with st.spinner("評価中..."):
                    try:
                        cached_eval = cache.get_evaluation(manual_title)
                        if cached_eval:
                            evaluation = cached_eval
                        else:
                            evaluation = title_evaluator.evaluate_title(manual_title)
                            cache.set_evaluation(manual_title, evaluation)
                        st.session_state.generated_titles.append(
                            GeneratedTitle(title=manual_title, evaluation=evaluation)
                        )
                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")
    
    # Step 3: 見出し生成
    if st.session_state.selected_title:
        st.header("Step 3: 見出し生成")
        
        if st.button("見出しを生成", key="generate_headlines"):
            with st.spinner("見出しを生成中..."):
                try:
                    st.session_state.headlines = headline_generator.generate_headlines(
                        st.session_state.selected_title
                    )
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
        
        if st.session_state.headlines:
            st.subheader("選択されたタイトル")
            selected_title_eval = next(
                (t.evaluation for t in st.session_state.generated_titles 
                 if t.title == st.session_state.selected_title), 
                None
            )
            
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.write(st.session_state.selected_title)
            if selected_title_eval:
                with cols[1]:
                    st.metric("集客速度", f"{selected_title_eval.speed:.1f}")
                with cols[2]:
                    grade_color = grade_colors.get(selected_title_eval.grade, "gray")
                    st.markdown(
                        f'<p style="color: {grade_color}; font-weight: bold; text-align: center;">評価: {selected_title_eval.grade}</p>',
                        unsafe_allow_html=True
                    )
            
            # 見出しの表示
            st.subheader("生成された見出し")
            cols = st.columns(3)
            with cols[0]:
                st.markdown("### 背景")
                st.write(st.session_state.headlines["background"])
            with cols[1]:
                st.markdown("### 課題")
                st.write(st.session_state.headlines["problem"])
            with cols[2]:
                st.markdown("### 解決策")
                st.write(st.session_state.headlines["solution"])

if __name__ == "__main__":
    main()
