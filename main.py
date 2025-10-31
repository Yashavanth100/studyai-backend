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

# Initialize Gemini AI
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key="AIzaSyB_MWOSd39SvbE6sc2ffbK0kRCYX4FaF8Y",  # 🔑 Replace this
)

class Question(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to StudyAI Backend!"}

@app.post("/ask")
async def ask_question(q: Question):
    try:
        # 🧠 More structured and Markdown-friendly prompt
        system_prompt = """
You are a structured and educational AI tutor.

Always format your answer **clearly** in Markdown with proper line breaks and bold headers. 
Use this exact format — each section must start on a new line.

**Topic:** <Main concept>

**Key Points:**
1. <Short key point 1>
2. <Short key point 2>
3. <Short key point 3>

**Example:**
<Give one short, realistic example with proper formatting and Markdown code block if needed.>

Be concise (max 150 words) and easy to read for students.
Add `\n\n` line breaks where needed.
"""

        prompt = ChatPromptTemplate.from_template(
            f"{system_prompt}\n\nQuestion: {{question}}"
        )

        prompt_text = prompt.format_prompt(question=q.question).to_string()

        # Run the model
        response = llm.invoke(prompt_text)

        # ✨ Ensure proper spacing and readability
        formatted_answer = (
            response.content.replace("**Key Points:**", "\n\n**Key Points:**")
            .replace("**Example:**", "\n\n**Example:**")
            .replace("1.", "\n1.")
            .replace("2.", "\n2.")
            .replace("3.", "\n3.")
            .strip()
        )

        return {"answer": formatted_answer}

    except Exception as e:
        return {"error": str(e)}
