import streamlit as st
from rag_bk.bk_logging import langsmith
from dotenv import load_dotenv
from rag_bk.modules.handler import stream_handler
# from sidebar import show_sidebar
from st_function import print_messages, add_message
from rag_bk.bk_messages import random_uuid

# API KEY 정보로드
api_key = st.secrets["OPENAI_API_KEY"]

# 프로젝트 이름
langsmith("챗봇상담")

st.title("The Health Curator 💬")
 
# 초기화 버튼 추가
if st.button("대화내용 초기화"):
    st.session_state["messages"] = []  # 대화 정보 지우기
    st.session_state["thread_id"] = random_uuid()  # 사용자정보 기억 지우기

# 대화기록을 저장하기 위한 용도로 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ReAct Agent 초기화
if "react_agent" not in st.session_state:
    st.session_state["react_agent"] = None

# 사이드바 생성(sidebar.py로 생성)
# show_sidebar()

import streamlit as st
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rag_bk.modules.tools import WebSearchTool, retriever_tool
from rag_bk.modules.agent import create_agent_executor
from rag_bk.bk_messages import random_uuid
# step 1 LLM을 통한 페르소나를 부여 프롬프트 생성하기
# gen_prom.yaml 로드(개인정보에 맞는 페르소나를 부여하기 위한 프롬프트)
loaded_prompt = load_prompt("prompts/gen_prom.yaml", encoding="utf-8")

# 최종 페르소나 부여 프롬프트 생성
final_template = (
    f"{loaded_prompt.template}, "  # 사용자 상담내용 + 개인정보에 맞는 페르소나를 부여하기 위한 프롬프트
)

# 페르소나 생성 LLM 호출 (첫 번째 체인)
# prompt1 = PromptTemplate.from_template(template=final_template)
# llm = ChatOpenAI(api_key=api_key, model_name="gpt-4o", temperature=0.5)
# chain1 = prompt1 | llm | StrOutputParser()
# st.session_state["new_prompt"] = chain1.invoke(
#     ""
# )

st.session_state["new_prompt"] = final_template

# step 2 생성된 페르소나를 이용한 최종 답변 LLM 생성
tool1 = retriever_tool()  # pdf_search

# WebSearchTool 인스턴스 생성
tool2 = WebSearchTool().create()  # web_search

# 리액트형 에이전트 생성
st.session_state["react_agent"] = create_agent_executor(
    model_name="gpt-4o-mini",
    tools=[tool1, tool2],
)

# 고유 스레드 ID(랜덤으로 지어주기 -> 대화 기억용도 -> 대화내용 초기화하면 이것도 초기화)
st.session_state["thread_id"] = random_uuid()

# 이전 대화 기록 출력(st_function.py로 생성)
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # Config 설정
    agent = st.session_state["react_agent"]
    if agent is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        # 사용자 메시지 출력
        st.chat_message("user").write(user_input)

        # 챗봇 응답 (스트리밍 + 이미지 함께 출력)
        with st.chat_message("assistant"):
            col1, col2 = st.columns([1, 9])

            with col1:
                st.image(
                    "https://github.com/jhyeok2841/health-curator-chatbot/blob/main/The_Health_Curator.png?raw=true",
                    width=50,
                )

            with col2:
                container = st.empty()  # 여기로 스트리밍 응답이 실시간 출력됨

        # 실제 응답 처리 (stream_handler가 container에 streaming 출력)
        container_messages, tool_args, agent_answer = stream_handler(
            container,
            agent,
            {"messages": [("human", user_input)]},
            config,
        )

        # 메시지 기록 저장
        add_message("user", user_input)
        for tool_arg in tool_args:
            add_message(
                "assistant",
                tool_arg["tool_result"],
                "tool_result",
                tool_arg["tool_name"],
            )
        add_message("assistant", agent_answer)

    else:
        warning_msg.warning("개인정보 입력을 완료해주세요.")
