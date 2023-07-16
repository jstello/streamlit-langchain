import streamlit as st
from streamlit import message

# Create a sidebar with some information about your chatbot.
with st.sidebar:
  st.title("My Chatbot")
  st.text("This is a simple chatbot that I built using Streamlit.")

# Create a session state to store the chat history.
session_state.chat_history = []

# Create the app layout.
st.markdown("<h2>Chat with me!</h2>")
st.text_input("Enter your message:")

# Get the user input.
user_input = st.text_input("Enter your message:")

# Generate a response using the chatbot's model.
response = message(user_input)

# Display the response to the user.
st.markdown(f"Bot: {response}")

# Append the chat history to the session state.
session_state.chat_history.append((user_input, response))

# Display the chat history to the user.
st.markdown("<h3>Chat history</h3>")
for chat_message in session_state.chat_history:
  st.markdown(f"User: {chat_message[0]}")
  st.markdown(f"Bot: {chat_message[1]}")
