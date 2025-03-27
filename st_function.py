from attr import dataclass
import streamlit as st
from langchain_core.messages.chat import ChatMessage
# from rag_bk.modules.handler import format_search_result


# 클레스 타입 지정 -> web_search시 답변을 이쁘게 만들기 위함.
@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str


# 이전 대화를 출력
def print_messages():
    for message in st.session_state["messages"]:

        # 그냥 일반 대화라면
        if message.msg_type == "text":
            st.chat_message(message.chat_message.role).write(
                message.chat_message.content
            )
        # tool을 사용한 경우
        elif message.msg_type == "tool_result":
            # tool이 web_search 이면 아래 format으로 답변
            if message.tool_name == "web_search":
                with st.expander(f"✅ {message.tool_name}"): 
                    st.markdown(message.chat_message.content)
            # tool이 pdf_search 이면 아래 format으로 답변
            elif message.tool_name == "pdf_search":
                with st.expander(f"✅ {message.tool_name}"):
                    st.chat_message(message.chat_message.role).write(
                        message.chat_message.content
                    )
            # else:
            #     st.chat_message(message.chat_message.role).write(
            #         message.chat_message.content
            #     )


# add_message도 세가지로 나눠서 답변을 저장하도록 저장
# 새로운 메시지를 추가
def add_message(role, message, msg_type="text", tool_name=""):
    if msg_type == "text":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role=role, content=message),
                msg_type="text",
                tool_name=tool_name,
            )
        )
    elif msg_type == "tool_result":
        if tool_name == "web_search":
            st.session_state["messages"].append(
                ChatMessageWithType(
                    chat_message=ChatMessage(
                        role="assistant", content=message#format_search_result(message)
                    ),
                    msg_type="tool_result",
                    tool_name=tool_name,
                )
            )
        elif msg_type == "pdf_search":
            st.session_state["messages"].append(
                ChatMessageWithType(
                    chat_message=ChatMessage(role=role, content=message),
                    msg_type="tool_result",
                    tool_name=tool_name,
                )
            )
