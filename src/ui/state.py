import streamlit as st
from typing import Optional, List
from ..models import GeneratedTitle, HeadlineSet
from ..utils.prompts import (
    TITLE_GENERATION_PROMPT,
    HEADLINE_GENERATION_PROMPT,
    BODY_GENERATION_PROMPT
)

def init_session_state():
    if 'generated_titles' not in st.session_state:
        st.session_state.generated_titles: List[GeneratedTitle] = []

    if 'selected_title' not in st.session_state:
        st.session_state.selected_title: Optional[str] = None

    if 'selected_title_for_headline' not in st.session_state:
        st.session_state.selected_title_for_headline: Optional[str] = None

    if 'selected_category' not in st.session_state:
        st.session_state.selected_category: Optional[str] = None

    if 'headlines' not in st.session_state:
        st.session_state.headlines: Optional[HeadlineSet] = None

    if 'title_cache' not in st.session_state:
        st.session_state.title_cache = {}

    if 'seminar_data' not in st.session_state:
        st.session_state.seminar_data = None

    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None

    if 'available_categories' not in st.session_state:
        st.session_state.available_categories: List[str] = []

    if 'extracted_content' not in st.session_state:
        st.session_state.extracted_content = {}

    if 'title_prompt' not in st.session_state:
        st.session_state.title_prompt = TITLE_GENERATION_PROMPT

    if 'headline_prompt' not in st.session_state:
        st.session_state.headline_prompt = HEADLINE_GENERATION_PROMPT

    if 'body_prompt' not in st.session_state:
        st.session_state.body_prompt = BODY_GENERATION_PROMPT

    if 'generated_body' not in st.session_state:
        st.session_state.generated_body: Optional[str] = None

    if 'manual_headlines' not in st.session_state:
        st.session_state.manual_headlines: Optional[HeadlineSet] = None