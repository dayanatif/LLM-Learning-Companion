import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define prompts
general_prompt = PromptTemplate(
    template="""You are an AI tutor for undergraduates. Explain the following question step by step in simple terms. Encourage reasoning before giving the final answer. If needed, provide clear and commented Python examples.

Question: {question}
""",
    input_variables=["question"]
)

code_prompt = PromptTemplate(
    template="""You are an AI tutor. A student uploaded some code. Your job is to:
1. Explain what the code does step by step in simple terms.
2. Point out any bugs or inefficiencies.
3. Suggest improvements (with corrected Python code if needed).

Here is the code:

{code}
""",
    input_variables=["code"]
)

general_chain = general_prompt | llm | StrOutputParser()
code_chain = code_prompt | llm | StrOutputParser()

# Streamlit UI
st.title("📘 AI-Powered Learning Companion")
st.write("Ask questions OR upload your code to get explanations with Gemini LLM.")

tab1, tab2 = st.tabs(["💡 Ask a Question", "💻 Upload Code"])

with tab1:
    user_input = st.text_area("Enter your question:")
    if st.button("Get Explanation", key="q_btn"):
        if user_input.strip():
            with st.spinner("Generating explanation..."):
                response = general_chain.invoke({"question": user_input})
                st.markdown("### 📖 Step-by-Step Explanation")
                st.write(response)
        else:
            st.warning("Please enter a question!")

with tab2:
    uploaded_file = st.file_uploader("Upload a Python code file", type=["py", "txt"])
    if uploaded_file is not None:
        code_content = uploaded_file.read().decode("utf-8")
        st.code(code_content, language="python")
        
        if st.button("Explain Code", key="c_btn"):
            with st.spinner("Analyzing code..."):
                response = code_chain.invoke({"code": code_content})
                st.markdown("### 📝 Code Explanation & Suggestions")
                st.write(response)
