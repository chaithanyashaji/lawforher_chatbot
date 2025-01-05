from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
TOGETHER_AI_API = os.getenv("TOGETHER_AI")

# Streamlit Page Config
st.set_page_config(page_title="Law4her")
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image(
        "https://res.cloudinary.com/dzzhbgbnp/image/upload/v1736073326/lawforher_logo1_yznqxr.png"
    )

st.markdown(
    """
    <style>
   div.stButton > button:first-child {
    background-color: #ffffff; /* White background */
    color: #000000; /* Black text */
    border: 1px solid #000000; /* Optional: Add a black border */
}

div.stButton > button:active {
    background-color: #e0e0e0; /* Slightly darker white for active state */
    color: #000000; /* Black text remains the same */
}

    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Reset Conversation
def reset_conversation():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]
    st.session_state.memory.clear()

# Initialize chat messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you?"}]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True
    )

# Load embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"},
)

# Enable dangerous deserialization (safe only if the file is trusted and created by you)
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2, "max_length": 512})

prompt_template = """<s>[INST]As a legal chatbot specializing in the Indian Penal Code, provide a concise and accurate answer based on the given context. Avoid unnecessary details or unrelated content. Only respond if the answer can be derived from the provided context; otherwise, say "The information is not available in the provided context." 
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# Initialize the Together API
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=TOGETHER_AI_API,
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# User input
input_prompt = st.chat_input("Ask a legal question about the Indian Penal Code")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            try:
                # Pass the user question
                result = qa.invoke(input=input_prompt)
                full_response = result.get("answer", "")

                # Ensure the answer is a string
                if isinstance(full_response, list):
                    full_response = " ".join(full_response)
                elif not isinstance(full_response, str):
                    full_response = str(full_response)

                # Display the response
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.write(full_response)

            except Exception as e:
                st.error(f"Error occurred: {e}")

# Add reset button
st.button("Reset All Chat 🗑", on_click=reset_conversation)
