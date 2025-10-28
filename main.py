from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini AI with correct model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=""  # <-- Add your real Google API key here
)

# Pydantic model for request body
class Question(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to StudyAI Backend!"}

@app.post("/ask")
async def ask_question(q: Question):
    try:
        # 🧠 System prompt to enforce structured format
        system_prompt = """
You are a smart, structured, and helpful AI tutor.
Always answer in a clean, note-style format with this structure:

**Topic:** <Main concept or definition>

**Key Points:**
1. <Key Point 1>
2. <Key Point 2>
3. <Key Point 3>

**Example:** <Short, real-world example or use case>

Keep it concise, educational, and easy to read for students.
"""

        # Create chat prompt using LangChain
        prompt = ChatPromptTemplate.from_template(
            f"{system_prompt}\n\nQuestion: {{question}}"
        )
        prompt_text = prompt.format_prompt(question=q.question).to_string()

        # Call Gemini model
        response = llm.invoke(prompt_text)

        # Return structured formatted content
        return {"answer": response.content.strip()}

    except Exception as e:
        # Return error for debugging
        return {"error": str(e)}
