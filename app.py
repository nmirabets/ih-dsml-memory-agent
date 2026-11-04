########################################################
# 1. Import the necessary libraries
########################################################
import streamlit as st
from agent import run_agent
from prompt import get_system_prompt

########################################################
# 2. Set the page config
########################################################
st.set_page_config(
    page_title="🧠 Memory AI Agent",
    page_icon=":material/chat_bubble_outline:",
    layout="centered",
    initial_sidebar_state="collapsed",
)
# Title
st.title("🧠 Memory AI Agent")

########################################################
# 3. Initialize the conversation history    
########################################################
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": get_system_prompt("")},
        {"role": "assistant", "content": "How can I help you?"}
    ]

########################################################
# 4. Display the conversation history
########################################################
for msg in st.session_state.messages[1:]:
    st.chat_message(msg["role"]).write(msg["content"])

########################################################
# 5. Send a new message
########################################################
prompt = st.chat_input()

if prompt:
    # Update system prompt with memories
    st.session_state["messages"][0]["content"] = get_system_prompt(prompt)

    # Add the user's message to the conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Invoke the model with the conversation history
    response = run_agent(list(st.session_state.messages))

    # Add the assistant's response to the conversation
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

