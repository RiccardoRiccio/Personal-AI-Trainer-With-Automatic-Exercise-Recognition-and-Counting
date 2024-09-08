import os
from dotenv import load_dotenv
import streamlit as st
from typing import Literal
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

def initialize_session_state():
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )

        # Insert propt as knowledge base to the chatbot to behave as an AI personal Trainer
        conversation_memory = ConversationSummaryMemory(llm=llm)
        conversation_memory.save_context({"human": ""}, {"ai": "You are a chatbot inserted in a web app that uses AI to classify and count the repetitions of home exercises. Act as an expert in fitness and respond to the user as their personal AI trainer."})
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=conversation_memory,
        )

def on_click_callback():
    human_prompt = st.session_state.get('human_prompt', '')
    if human_prompt:
        llm_response = st.session_state.conversation.run(human_prompt)
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        st.session_state.token_count += len(llm_response.split())  # Example token approximation

        # Clear the text input field after submitting the message
        st.session_state.human_prompt = ""

def chat_ui():
    initialize_session_state()
    st.title("Ask me anything about Fitness 🤖")

    # Define custom CSS style for message
    custom_css = """
        <style>
            .chat-bubble {
                background-color: #f1f0f0;
                padding: 10px 15px;
                border-radius: 20px;
                margin-bottom: 10px;
                max-width: 70%;
                display: inline-block;
                word-wrap: break-word; /* Add this line to wrap long words */
                max-width: 30%; /* Adjusted width */
                color: black; /* Set text color to black */
            }
            .user-bubble {
                background-color: #d0f0f0;
                align-self: flex-end;
                margin-left: 500px;
                color: black; /* Set text color to black */
            }
            .ai-bubble {
                background-color: #f0f0f0;
                align-self: flex-start;
                color: black; /* Set text color to black */
            }
        </style>
    """

    # Display custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")

    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
            <div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
                <div class="chat-bubble {'user-bubble' if chat.origin == 'human' else 'ai-bubble'}">
                    {chat.message}
                </div>
            </div>
            """
            st.markdown(div, unsafe_allow_html=True)

    with prompt_placeholder:
        st.text_input("Chat", key="human_prompt")
        st.form_submit_button("Submit", on_click=on_click_callback)

    # st.caption(f"Used {st.session_state.token_count} tokens")

if __name__ == "__main__":
    chat_ui()
