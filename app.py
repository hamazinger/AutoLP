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

[... rest of the code ...]