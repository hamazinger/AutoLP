from typing import Optional
import requests
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract
from ..models.data_classes import WebContent

class URLContentExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0'
        }

    def extract_with_trafilatura(self, url: str) -> Optional[WebContent]:
        try:
            downloaded = fetch_url(url)
            if downloaded is None:
                return WebContent(
                    title="",
                    description="",
                    main_content="",
                    error="URLからのコンテンツ取得に失敗しました"
                )

            content = extract(downloaded, include_comments=False, include_tables=False)
            if content is None:
                return WebContent(
                    title="",
                    description="",
                    main_content="",
                    error="コンテンツの抽出に失敗しました"
                )

            soup = BeautifulSoup(downloaded, 'html.parser')
            title = soup.title.string if soup.title else ""
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ""

            return WebContent(
                title=title,
                description=description,
                main_content=content
            )
        except Exception as e:
            return WebContent(
                title="",
                description="",
                main_content="",
                error=f"エラーが発生しました: {str(e)}"
            )