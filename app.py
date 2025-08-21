import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import re

# Load .env file for OpenAI key
load_dotenv()

# Get OpenAI API key from environment or Streamlit secrets
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

if not openai_api_key:
    st.error("🚨 OpenAI API key not found! Please add it to your Streamlit secrets.")
    st.info("Go to your Streamlit app settings and add OPENAI_API_KEY to secrets.")
    st.stop()

# Configure OpenAI LLM
llm = OpenAI(
    api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1000
)
Settings.llm = llm

# Streamlit page config
st.set_page_config(
    page_title="Teach Assist", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS based on your design mockups and color palette
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Color Palette from your design */
:root {
    --color-primary: #274C77;      /* Dark blue */
    --color-secondary: #6096BA;    /* Medium blue */
    --color-accent: #A3CEF1;       /* Light blue */
    --color-orange: #FB5012;       /* Orange accent */
    --color-light: #E7ECEF;        /* Light gray */
    --color-white: #FFFFFF;
    --color-text: #1a1a1a;
    --color-text-light: #666666;
}

/* Hide Streamlit default elements */
.stApp > header {display: none;}
.stApp > .main > div {padding-top: 0rem;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Global styling */
* {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.main-container {
    background: transparent;
}

/* Header styling */
.app-header {
    background: var(--color-white);
    padding: 2rem 3rem;
    text-align: center;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 0;
}

.app-title {
    font-size: 3rem;
    font-weight: 700;
    color: #1a1a1a !important;
    margin: 0;
    letter-spacing: -1px;
}

.app-subtitle {
    font-size: 1.1rem;
    color: var(--color-text-light);
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Setup screen styling */
.setup-container {
    max-width: 1200px;
    margin: 3rem auto;
    padding: 0 2rem;
}

.setup-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}

.setup-card {
    background: var(--color-white);
    border-radius: 12px;
    padding: 2.5rem;
    border: 2px solid #e5e7eb;
    transition: all 0.3s ease;
    text-align: center;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.setup-card:hover {
    border-color: var(--color-secondary);
    box-shadow: 0 10px 30px rgba(39, 76, 119, 0.1);
    transform: translateY(-2px);
}

.setup-card-large {
    grid-column: 1 / -1;
    min-height: 150px;
}

.setup-card h3 {
    color: var(--color-primary);
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.setup-card p {
    color: var(--color-text-light);
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* Chat interface styling */
.chat-interface {
    display: flex;
    height: 100vh;
    max-height: 100vh;
    overflow: hidden;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
}

.chat-panel {
    flex: 1.2;
    background: var(--color-white);
    border-right: 1px solid #e5e7eb;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}

.content-panel {
    flex: 0.8;
    background: var(--color-light);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    position: relative;
}

/* Chat header */
.chat-header {
    padding: 1.5rem 2rem;
    border-bottom: 1px solid #e5e7eb;
    background: var(--color-white);
    flex-shrink: 0;
}

.chat-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
}

.chat-subtitle {
    font-size: 0.9rem;
    color: var(--color-text-light);
    margin-top: 0.3rem;
}

/* Quick actions */
.quick-actions {
    padding: 1rem 2rem;
    border-bottom: 1px solid #e5e7eb;
    background: #fafbfc;
}

.quick-actions h4 {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--color-text);
    margin: 0 0 0.8rem 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.action-buttons {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
}

/* Chat messages */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 2rem;
    background: var(--color-white);
}

.message {
    margin-bottom: 1.5rem;
    animation: fadeInUp 0.3s ease;
}

.message-user {
    display: flex;
    justify-content: flex-end;
}

.message-ai {
    display: flex;
    justify-content: flex-start;
}

.message-bubble {
    max-width: 80%;
    padding: 1rem 1.2rem;
    border-radius: 12px;
    font-size: 0.95rem;
    line-height: 1.5;
}

.message-bubble-user {
    background: var(--color-primary);
    color: var(--color-white);
    border-bottom-right-radius: 4px;
}

.message-bubble-ai {
    background: #f8fafc;
    color: var(--color-text);
    border: 1px solid #e5e7eb;
    border-bottom-left-radius: 4px;
}

.message-header {
    font-weight: 600;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}

/* Chat input */
.chat-input {
    padding: 1.5rem 2rem;
    border-top: 1px solid #e5e7eb;
    background: var(--color-white);
    flex-shrink: 0;
}

/* Content panel styling */
.content-header {
    padding: 1.5rem 2rem;
    border-bottom: 1px solid #e5e7eb;
    background: var(--color-white);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.content-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--color-primary);
    margin: 0;
}

.content-area {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    background: var(--color-light);
}

.content-display {
    background: var(--color-white);
    border-radius: 12px;
    padding: 2rem;
    border: 1px solid #e5e7eb;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--color-text);
    min-height: 400px;
}

.content-placeholder {
    text-align: center;
    color: var(--color-text-light);
    font-size: 1rem;
    padding: 3rem 2rem;
}

/* Button styling */
.stButton > button {
    background: var(--color-secondary) !important;
    color: var(--color-white) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.stButton > button:hover {
    background: var(--color-primary) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Orange accent buttons */
.stButton > button[kind="primary"] {
    background: var(--color-orange) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #e8460f !important;
}

/* Form styling */
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.95rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--color-secondary) !important;
    box-shadow: 0 0 0 2px rgba(96, 150, 186, 0.2) !important;
}

.stTextArea > div > div > textarea {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    padding: 0.8rem 1rem !important;
    font-size: 0.95rem !important;
}

/* Welcome message styling */
.welcome-message {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--color-text-light);
}

.welcome-message h3 {
    color: var(--color-primary);
    margin-bottom: 1rem;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive design */
@media (max-width: 1024px) {
    .chat-interface {
        flex-direction: column;
        position: relative;
        height: auto;
    }
    
    .chat-panel {
        flex: none;
        height: auto;
        max-height: none;
    }
    
    .content-panel {
        flex: none;
        height: auto;
        max-height: none;
        border-top: 1px solid #e5e7eb;
        border-right: none;
    }
    
    .setup-grid {
        grid-template-columns: 1fr;
    }
}

/* Hide streamlit elements */
.stDeployButton {display: none;}
.stDecoration {display: none;}
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_template(filename):
    template_path = os.path.join("templates", filename)
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return f"Template file '{filename}' not found. Please ensure templates folder exists."

def create_pdf_from_content(content, title="Teach Assist Content"):
    """Create a PDF from the given content"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    content_style = ParagraphStyle(
        'ContentStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leftIndent=0,
        rightIndent=0,
    )
    
    story = []
    title_para = Paragraph(title, title_style)
    story.append(title_para)
    
    date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    date_para = Paragraph(f"<i>Generated on {date_str}</i>", styles['Normal'])
    story.append(date_para)
    story.append(Spacer(1, 20))
    
    content_lines = content.split('\n')
    for line in content_lines:
        if line.strip():
            formatted_line = line
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_line)
            formatted_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_line)
            formatted_line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', formatted_line)
            
            if line.startswith('###'):
                formatted_line = f"<b>{line.replace('###', '').strip()}</b>"
            elif line.startswith('##'):
                formatted_line = f"<b><font size=14>{line.replace('##', '').strip()}</font></b>"
            elif line.startswith('#'):
                formatted_line = f"<b><font size=16>{line.replace('#', '').strip()}</font></b>"
            
            para = Paragraph(formatted_line, content_style)
            story.append(para)
        else:
            story.append(Spacer(1, 6))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# File processing
input_dir = "uploaded_input"
os.makedirs(input_dir, exist_ok=True)

def load_course_context():
    if os.path.exists(input_dir) and os.listdir(input_dir):
        file_path = input_dir
    else:
        default_path = "input_files"
        if os.path.exists(default_path) and os.listdir(default_path):
            file_path = default_path
        else:
            return None, None
    
    try:
        documents = SimpleDirectoryReader(file_path).load_data()
        course_context = "\n\n".join([doc.text for doc in documents])
        return course_context, len(documents)
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")
        return None, None

course_context, num_docs = load_course_context()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_query" not in st.session_state:
    st.session_state.processing_query = False
if "current_content" not in st.session_state:
    st.session_state.current_content = None
if "content_title" not in st.session_state:
    st.session_state.content_title = ""

# Main app layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)

if course_context is None:
    # Setup Screen (Based on your mockup design)
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Teach Assist</h1>
        <p class="app-subtitle">AI-powered teaching companion that helps you create engaging lesson plans, quizzes, and educational content</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="setup-container">', unsafe_allow_html=True)
    
    # Setup grid based on your mockup
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="setup-card">
            <h3>📄 Download Templates</h3>
            <p>Get structured templates for curriculum and pedagogy to jumpstart your course setup</p>
        </div>
        """, unsafe_allow_html=True)
        
        curriculum_data = load_template("curriculum_template.md")
        st.download_button(
            label="📄 Curriculum Template",
            data=curriculum_data,
            file_name="curriculum_template.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        pedagogy_data = load_template("pedagogy_template.md")
        st.download_button(
            label="📚 Pedagogy Template",
            data=pedagogy_data,
            file_name="pedagogy_template.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        st.markdown("""
        <div class="setup-card">
            <h3>📤 Upload Your Files</h3>
            <p>Upload your completed curriculum and pedagogy files to get personalized assistance</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload curriculum.md and pedagogy.md files",
            type=["md", "txt"],
            accept_multiple_files=True,
            key="main_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            for file in os.listdir(input_dir):
                os.remove(os.path.join(input_dir, file))
            
            for file in uploaded_files:
                with open(os.path.join(input_dir, file.name), "wb") as f:
                    f.write(file.read())
            
            st.success("✅ Files uploaded successfully!")
            if st.button("🚀 Start Teaching", type="primary", use_container_width=True):
                st.rerun()
    
    # AI Generation section (full width)
    st.markdown("""
    <div class="setup-card setup-card-large">
        <h3>🤖 Generate Files for You</h3>
        <p>Describe your course and let AI create comprehensive curriculum and pedagogy files</p>
    </div>
    """, unsafe_allow_html=True)
    
    course_description = st.text_area(
        "Describe your course:",
        placeholder="e.g., A 5-day intermediate course on Retrieval Augmented Generation (RAG) for software engineers with Python experience...",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("🚀 Generate Course Files", type="primary", use_container_width=True):
        if course_description.strip():
            with st.spinner("🤖 Creating your course files..."):
                try:
                    # Generate curriculum
                    curriculum_prompt = f"""Create a detailed curriculum.md file for this course: {course_description}
                    
                    Format it as a proper markdown file with sections for:
                    - Course Title
                    - Course Description  
                    - Learning Objectives
                    - Prerequisites
                    - Course Modules (with topics and key concepts)
                    - Assessment Methods
                    - Resources
                    - Course Schedule
                    
                    Make it comprehensive and well-structured."""
                    
                    curriculum_messages = [
                        ChatMessage(role="system", content="You are an expert curriculum designer. Create detailed, practical curriculum files."),
                        ChatMessage(role="user", content=curriculum_prompt)
                    ]
                    
                    curriculum_response = llm.chat(curriculum_messages)
                    
                    # Generate pedagogy
                    pedagogy_prompt = f"""Create a detailed pedagogy.md file for this course: {course_description}
                    
                    Format it as a proper markdown file with sections for:
                    - Teaching Philosophy
                    - Target Audience
                    - Learning Styles Accommodation
                    - Teaching Methods
                    - Assessment Strategies
                    - Engagement Techniques
                    - Technology Integration
                    - Differentiation Strategies
                    - Classroom Management
                    - Support Resources
                    
                    Make it practical and actionable for instructors."""
                    
                    pedagogy_messages = [
                        ChatMessage(role="system", content="You are an expert in educational pedagogy. Create detailed, practical teaching guides."),
                        ChatMessage(role="user", content=pedagogy_prompt)
                    ]
                    
                    pedagogy_response = llm.chat(pedagogy_messages)
                    
                    # Save generated files
                    with open(os.path.join(input_dir, "curriculum.md"), "w", encoding="utf-8") as f:
                        f.write(curriculum_response.message.content)
                    
                    with open(os.path.join(input_dir, "pedagogy.md"), "w", encoding="utf-8") as f:
                        f.write(pedagogy_response.message.content)
                    
                    st.success("✅ Course files generated successfully!")
                    if st.button("🚀 Start Teaching", type="primary", use_container_width=True, key="start_after_gen"):
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error generating files: {str(e)}. Please check your OpenAI API key.")
        else:
            st.warning("⚠️ Please describe your course first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Chat Interface (Based on your two-panel mockup)
    st.markdown('<div class="chat-interface">', unsafe_allow_html=True)
    
    # Left Panel - Chat
    col_chat, col_content = st.columns([1.2, 0.8])
    
    with col_chat:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    
    # Chat Header
    st.markdown(f"""
    <div class="chat-header">
        <h2 class="chat-title">Teach Assist</h2>
        <p class="chat-subtitle">📚 {num_docs} course documents loaded</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("""
    <div class="quick-actions">
        <h4>💡 Quick Actions</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 Lesson Plan", disabled=st.session_state.processing_query, use_container_width=True, key="btn_lesson"):
            st.session_state.next_query = "Create a detailed lesson plan for module 1"
            st.rerun()
        if st.button("🎯 Activities", disabled=st.session_state.processing_query, use_container_width=True, key="btn_activities"):
            st.session_state.next_query = "Suggest interactive activities to engage students"
            st.rerun()
    with col2:
        if st.button("❓ Quiz Questions", disabled=st.session_state.processing_query, use_container_width=True, key="btn_quiz"):
            st.session_state.next_query = "Generate 10 quiz questions for this module"
            st.rerun()
        if st.button("📝 Assignment", disabled=st.session_state.processing_query, use_container_width=True, key="btn_assignment"):
            st.session_state.next_query = "Create an assignment for this module"
            st.rerun()
    
    # Chat Messages
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for i, (sender, msg) in enumerate(st.session_state.chat_history):
            if sender == "user":
                st.markdown(f"""
                <div class="message message-user">
                    <div class="message-bubble message-bubble-user">
                        <div class="message-header">You</div>
                        {msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                summary = msg[:120] + "..." if len(msg) > 120 else msg
                st.markdown(f"""
                <div class="message message-ai">
                    <div class="message-bubble message-bubble-ai">
                        <div class="message-header">AI Copilot</div>
                        {summary}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if len(msg) > 120:
                    if st.button("📄 View Full Content", key=f"view_{i}", use_container_width=True):
                        st.session_state.current_content = msg
                        st.session_state.content_title = f"Response to: {st.session_state.chat_history[i-1][1][:50]}..."
                        st.rerun()
    else:
        st.markdown("""
        <div class="welcome-message">
            <h3>👋 Welcome to Teach Assist!</h3>
            <p>I'm here to help you create amazing educational content. Try asking me to generate lesson plans, quizzes, or activities for your course.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat Input
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Type your message:",
                placeholder="Ask about lesson plans, quizzes, activities...",
                disabled=st.session_state.processing_query,
                label_visibility="collapsed"
            )
        with col_send:
            submit_button = st.form_submit_button(
                "Send",
                disabled=st.session_state.processing_query,
                use_container_width=True
            )
    
    # Settings button
    with st.popover("⚙️"):
        st.markdown("**Course Management:**")
        if st.button("🗑️ Clear Course Documents", use_container_width=True):
            if os.path.exists(input_dir):
                for file in os.listdir(input_dir):
                    try:
                        os.remove(os.path.join(input_dir, file))
                    except:
                        pass
            st.session_state.chat_history = []
            st.session_state.current_content = None
            st.session_state.content_title = ""
            st.session_state.processing_query = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Right Panel - Content Viewer (matches your mockup)
    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    
    # Content Header
    col_title, col_download = st.columns([4, 1])
    with col_title:
        if st.session_state.current_content:
            st.markdown(f'<h3 class="content-title">{st.session_state.content_title}</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 class="content-title">Output Content Generated by AI</h3>', unsafe_allow_html=True)
    
    with col_download:
        if st.session_state.current_content:
            try:
                pdf_buffer = create_pdf_from_content(
                    st.session_state.current_content, 
                    st.session_state.content_title
                )
                st.download_button(
                    "Download PDF",
                    data=pdf_buffer,
                    file_name=f"teach_assist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF Error: {str(e)}")
    
    # Content Area
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    
    if st.session_state.current_content:
        st.markdown(f"""
        <div class="content-display">
            {st.session_state.current_content.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="content-placeholder">
            <h3>📄 Content will appear here</h3>
            <p>When you generate lesson plans, quizzes, or other detailed content, it will be displayed in this panel for easy reading and export.</p>
            <p><strong>Features:</strong></p>
            <p>• Clean, formatted display<br>
            • PDF export functionality<br>
            • Easy copying and sharing</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle queued queries and form submission
    if "next_query" in st.session_state:
        user_input = st.session_state.next_query
        del st.session_state.next_query
        submit_button = True
    
    # Process query
    if submit_button and user_input and not st.session_state.processing_query:
        st.session_state.processing_query = True
        st.session_state.chat_history.append(("user", user_input))
        
        with st.spinner("🤖 Generating response..."):
            try:
                system_prompt = f"""You are Teach Assist, an AI-powered teaching companion designed to help instructors create engaging lesson plans, teaching materials, and educational content.

                COURSE CONTEXT:
                {course_context}
                
                Based on the curriculum and pedagogy information above, help the instructor with their request. 
                Be specific, practical, and reference the course content when relevant.
                Create detailed, actionable responses that instructors can use immediately.
                
                For longer content like lesson plans, quizzes, or assignments, provide comprehensive, well-structured responses with clear formatting."""
                
                messages = [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_input)
                ]
                
                response = llm.chat(messages)
                response_content = response.message.content
                
                st.session_state.chat_history.append(("ai", response_content))
                
                # Auto-display longer responses in content panel
                if len(response_content) > 150:
                    st.session_state.current_content = response_content
                    st.session_state.content_title = f"Response: {user_input[:50]}..."
                
                st.session_state.processing_query = False
                st.rerun()
                
            except Exception as e:
                st.session_state.chat_history.append(("ai", f"❌ Error: {str(e)}"))
                st.session_state.processing_query = False
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)