import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Configure page settings first
st.set_page_config(
    page_title="Career Path AI",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for dark theme
st.markdown("""
<style>
    body {
        background-color: #2f4f7f;
        color: #ffffff;
    }
    [data-testid=stSidebar] {
        background-color: #1a1d23;
        padding: 20px;
    }
    .user-message {
        background-color: #3a3d41;
        border-radius: 10px;
        padding: 15px;
    }
    .bot-message {
        background-color: #4a4d51;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Check for API Key
if not os.getenv("GEMINI_API_KEY"):
    st.error("🔑 Missing GEMINI_API_KEY in environment variables")
    st.stop()

# Configure Gemini model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Start the chat session with original instructions
chat_session = model.start_chat(
    history=[
        {"role": "user", "parts": [
            "You are a career assistant AI. Your purpose is to provide advice, guidance, and information strictly related to careers, jobs, professional development, and education. You will not answer questions outside of these topics.\n\nIf a user asks something unrelated to careers, politely guide them back to career-related discussions. Your responses should be clear, professional, and actionable.\n\nFocus on helping users with:\n\n- Career advice and planning\n- Resume writing and optimization\n- Job interview tips and preparation\n- Skills development and learning resources\n- Job search strategies\n- Industry trends and career growth opportunities\n\nIf a question is vague, ask clarifying questions to better assist the user."
        ]},
        {"role": "model", "parts": [
            "Understood. I'm ready to assist users with their career-related inquiries. Let the users begin!"
        ]}
    ]
)

# Initialize session state for chat history (only for display)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Define history file name
history_file = "chat_logs.txt"

# Sidebar: Career Tools and Chat History
with st.sidebar:
    st.header("🔍 Career Tools")

    if st.checkbox("📄 Resume Tips"):
        st.subheader("Resume Tips")
        st.markdown("""
        - Tailor your resume for each job.
        - Use numbers to show impact.
        - Keep it concise and use action verbs.
        - Highlight accomplishments, not just duties.
        - Proofread and make it ATS-friendly.
        """)
    if st.checkbox("🤝 Interview Prep"):
        st.subheader("Interview Prep")
        st.markdown("""
        - Research the company.
        - Use STAR method.
        - Dress well, and practice questions.
        - Prepare questions for them.
        """)
    if st.checkbox("📈 Career Growth"):
        st.subheader("Career Growth")
        st.markdown("""
        - Set goals and build new skills.
        - Network and seek feedback.
        - Be adaptable and create your brand.
        """)
    if st.checkbox("🛠 Skill Development"):
        st.subheader("Skill Development")
        st.markdown("""
        - Identify gaps and take online courses.
        - Practice and earn certifications.
        - Apply through real projects.
        """)
    if st.checkbox("🤝 Networking Tips"):
        st.subheader("Networking Tips")
        st.markdown("""
        - Build quality connections.
        - Use LinkedIn well.
        - Attend events and follow up.
        """)
    if st.checkbox("🔍 Job Search Strategies"):
        st.subheader("Job Search")
        st.markdown("""
        - Use job boards wisely.
        - Tailor applications.
        - Reach out to recruiters.
        """)

    st.markdown("---")
    st.header("📜 Chat History")

    # Clear History button
    if st.button("🧹 Clear History"):
        if os.path.exists(history_file):
            open(history_file, "w", encoding="utf-8").close()
            st.success("Chat history cleared.")

    # Display previous conversation blocks
    if os.path.exists(history_file):
        data = open(history_file, "r", encoding="utf-8").read().strip()
        if data:
            blocks = data.strip().split("-----\n")
            for i, block in enumerate(reversed(blocks)):
                if "User: " in block and "Bot: " in block:
                    user_line = next((line for line in block.splitlines() if line.startswith("User: ")), None)
                    bot_line = next((line for line in block.splitlines() if line.startswith("Bot: ")), None)
                    if user_line and bot_line:
                        user_msg = user_line.replace("User: ", "").strip()
                        bot_msg = bot_line.replace("Bot: ", "").strip()
                        with st.expander(f"🗂 Conversation {len(blocks)-i}", expanded=False):
                            st.markdown(f"**👤 You:** {user_msg}")
                            st.markdown(f"**🤖 Career Path AI:** {bot_msg}")
        else:
            st.write("No past history available.")
    else:
        st.write("No past history available.")

# Main interface
st.title("💬 Career Path AI")
st.caption("Your personal career development assistant")

# Load avatars
try:
    user_avatar = Image.open("images/user_avatar.png")
    bot_avatar = Image.open("images/bot_avatar.png")
except FileNotFoundError:
    user_avatar = None
    bot_avatar = None

# Chat container
chat_container = st.container()

# Input form
with st.form(key="user_input_form"):
    user_input = st.text_input("Type your career question...", key="input")
    col1, col2 = st.columns([4, 1])
    with col1:
        submit_button = st.form_submit_button(label="🚀 Send")
    with col2:
        clear_button = st.form_submit_button(label="🗑️ Clear Chat")

# Clear in-session chat
if clear_button:
    st.session_state.chat_history = []
    st.rerun()

# Process chat
if submit_button and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("🔍 Analyzing your query..."):
        try:
            response = chat_session.send_message(user_input)
            bot_response = response.text
        except Exception as e:
            bot_response = f"⚠️ Error: {str(e)}"
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    # Save chat to file with separator
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"Bot: {bot_response}\n")
        f.write("-----\n")

# Display chat
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar=user_avatar):
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar=bot_avatar):
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
                if "resume" in message["content"].lower():
                    st.success("Pro Tip: Always quantify achievements in your resume!")
                elif "interview" in message["content"].lower():
                    st.info("Remember: Practice STAR method for behavioral questions")
