import requests
import json

import streamlit as st

FLASK_URL = "http://127.0.0.1:5000/chatbot"

# Initialize ChatMessageHistory in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "user_input_temp" not in st.session_state:
    st.session_state.user_input_temp = ""

if "send_on_enter" not in st.session_state:
    st.session_state.send_on_enter = False

def submit():
    """
    Clear the input after submit
    """
    st.session_state.user_input_temp = st.session_state.user_input
    st.session_state.user_input = ''            # clear the visible input
    if st.session_state.user_input_temp != '':
        st.session_state.send_on_enter = True       # signal main loop to send
    else:
        st.session_state.send_on_enter = False

# ---- UI ----
st.set_page_config(page_title="Agent 聊天機器人 Demo", layout="wide")

st.title("💬 Chat with Conversational Agent")

# --- CSS for Slack-like scrollable chat window ---
st.markdown("""
    <style>
    .chat-box {
        height: 500px;               /* fixed height */
        overflow-y: auto;            /* vertical scroll */
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 10px;
        background-color: #fafafa;
    }
    .user-msg {
        color: #1d1d1d;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .ai-msg {
        color: #0a66c2;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.header("Chat History")
chat_container = st.container()
input_container = st.container()

with input_container:
    _ = st.text_input(
        "Type your message...", 
        key="user_input",   # Streamlit will store the text box’s current value in: st.session_state.user_input
        on_change=submit,
        # label_visibility="collapsed"  # hides the label above input
    )
    user_input = st.session_state.user_input_temp

    col1, col2 = st.columns([8, 1])
    with col2:
        send_button = st.button("Send")
    
    if send_button or st.session_state.send_on_enter:
        
        if user_input.strip() != "":
            # Add user message
            st.session_state.chat_history.append({"content": user_input, "role": "user"})
            
            # Send to Flask
            response = requests.post(
                FLASK_URL,
                json={"chat_history": st.session_state.chat_history}
            )

            # Add AI reply
            if response.status_code == 200:
                ai_reply = response.json().get("messages", "No reply from AI.")
                st.session_state.chat_history.append({"content": ai_reply[-1]['content'], "role": "ai"})
            else:
                st.session_state.chat_history.append(
                    {"content": "Error: Could not get AI reply.", "role": "ai"}
                )

with chat_container:
    chat_messages = ""
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            # st.markdown(f"**🧑 You:** {content}")
            chat_messages += f"<div class='user-msg'>🧑 <b>You:</b> {content}</div>"
        else:
            # st.markdown(f"**🤖 AI:** {content}")
            chat_messages += f"<div class='ai-msg'>🤖 <b>AI:</b> {content}</div>"

    st.markdown(f"<div class='chat-box'>{chat_messages}</div>", unsafe_allow_html=True)


    