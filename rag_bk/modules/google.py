import streamlit as st
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from typing import List, Optional
import json
import os

load_dotenv()
api_key = st.secrets["SERPER_API_KEY"]

class GoogleSearchInput(BaseModel):
    """Google 검색 도구의 입력 모델"""
    query: str = Field(description="검색할 쿼리")


def format_google_search_result(result: dict) -> str:
    """
    Google 검색 결과를 포맷팅하는 함수.

    Args:
        result (dict): 원본 검색 결과

    Returns:
        str: XML 형식으로 포맷된 검색 결과
    """
    title = json.dumps(result.get("title", ""), ensure_ascii=False)[1:-1]
    snippet = json.dumps(result.get("snippet", ""), ensure_ascii=False)[1:-1]
    url = result.get("link", "")

    return f"<document><title>{title}</title><url>{url}</url><content>{snippet}</content></document>"


class GoogleSearch(BaseTool):
    """
    Google 검색을 수행하는 도구 클래스
    """

    name: str = "google_web_search"
    description: str = (
        "A search engine that retrieves relevant results from Google. "
        "Useful for answering current event queries. "
        "Input should be a search query."
    )
    args_schema: type[BaseModel] = GoogleSearchInput
    client: GoogleSerperAPIWrapper = None
    max_results: int = 3

    def __init__(self, api_key: Optional[str] = None, max_results: int = 3):
        """
        GoogleSearch 클래스 초기화

        Args:
            api_key (str): Google Serper API 키
            max_results (int): 최대 검색 결과 수
        """
        super().__init__()
        if api_key is None:
            api_key = os.environ.get("SERPER_API_KEY", None)

        if api_key is None:
            raise ValueError("Google Serper API key is not set.")

        self.client = GoogleSerperAPIWrapper(api_key=api_key)
        self.max_results = max_results

    def _run(self, query: str) -> str:
        """검색 실행"""
        return self.search(query)

    def search(self, query: str) -> list:
        """
        Google 검색을 수행하고 결과를 반환합니다.
        """
        response = self.client.results(query)  # query만 전달

        # 검색 결과가 없는 경우 빈 리스트 반환
        if "organic" not in response:
            return []

        # 검색 결과를 필요한 개수(max_results)만큼 제한
        return [format_google_search_result(r) for r in response["organic"][:self.max_results]]


    def get_search_context(self, query: str, max_results: int = 5) -> str:
        """
        검색 결과를 JSON 형태로 반환하는 메서드

        Args:
            query (str): 검색 쿼리
            max_results (int): 최대 검색 결과 수

        Returns:
            str: 검색 컨텍스트 (JSON)
        """
        results = self.search(query, max_results)
        return json.dumps(results, ensure_ascii=False)
