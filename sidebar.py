import streamlit as st
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rag_bk.modules.tools import WebSearchTool, retriever_tool
from rag_bk.modules.agent import create_agent_executor
from rag_bk.bk_messages import random_uuid


# 사이드바 배치 함수화
def show_sidebar():
    # step 1 LLM을 통한 페르소나를 부여 프롬프트 생성하기
    # gen_prom.yaml 로드(개인정보에 맞는 페르소나를 부여하기 위한 프롬프트)
    loaded_prompt = load_prompt("prompts/gen_prom.yaml", encoding="utf-8")

    # 최종 페르소나 부여 프롬프트 생성
    final_template = (
        f"{loaded_prompt.template}, "  # 사용자 상담내용 + 개인정보에 맞는 페르소나를 부여하기 위한 프롬프트
    )

    # 페르소나 생성 LLM 호출 (첫 번째 체인)
    prompt1 = PromptTemplate.from_template(template=final_template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
    chain1 = prompt1 | llm | StrOutputParser()
    st.session_state["new_prompt"] = chain1.invoke(
        ""
    )  # 생성된 페르소나 저장

    # step 2 생성된 페르소나를 이용한 최종 답변 LLM 생성
    tool1 = retriever_tool()  # pdf_search

    # WebSearchTool 인스턴스 생성
    tool2 = WebSearchTool().create()  # web_search

    # 리액트형 에이전트 생성
    st.session_state["react_agent"] = create_agent_executor(
        model_name="gpt-4o",
        tools=[tool1, tool2],
    )
    # 고유 스레드 ID(랜덤으로 지어주기 -> 대화 기억용도 -> 대화내용 초기화하면 이것도 초기화)
    st.session_state["thread_id"] = random_uuid()