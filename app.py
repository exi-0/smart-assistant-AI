import os
import torch
import pandas as pd
from typing import List
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai


# ENV + MODEL SETUP

# ENV + MODEL SETUP (Optimized for Render)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Safe Gemini setup
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print("❌ Gemini init error:", e)

GEMINI_MODEL = "gemini-2.5-flash"

# Lazy-load embedding model (fast startup)
embedding_model = None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("⚡ Loading MiniLM model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

# Safe Pinecone setup (won’t crash app)
try:
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "employee-support-kb"
    existing_indexes = [index.name for index in pinecone.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    kb_index = pinecone.Index(INDEX_NAME)

except Exception as e:
    print("❌ Pinecone init error:", e)
    kb_index = None


#                DATA LOADING

DATA_PATH = "data/employee_data.xlsx"

def load_pdf_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_all_data():
    texts = []
    data_paths = {
        "complaint_csv": "data/complaint_knowledge.csv",
        "employee_xlsx": "data/employee_data.xlsx",
        "employee_pdf": "data/All_Employee_Report.pdf"
    }

    if os.path.exists(data_paths["complaint_csv"]):
        df_csv = pd.read_csv(data_paths["complaint_csv"])
        for _, row in df_csv.iterrows():
            texts.append(" | ".join(map(str, row.values)))

    if os.path.exists(data_paths["employee_xlsx"]):
        df_xlsx = pd.read_excel(data_paths["employee_xlsx"])
        for _, row in df_xlsx.iterrows():
            texts.append(" | ".join(map(str, row.values)))

    if os.path.exists(data_paths["employee_pdf"]):
        pdf_text = load_pdf_text(data_paths["employee_pdf"])
        pdf_chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
        texts.extend(pdf_chunks)

    return texts

def embed_and_index_texts(texts: List[str], batch_size: int = 100):
    if not texts:
        print("⚠️ No data found to index.")
        return

    ids = [f"doc_{i}" for i in range(len(texts))]
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()
    print(f"📤 Uploading {len(texts)} vectors to Pinecone...")

    for i in range(0, len(texts), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_vectors = embeddings[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        records = [
            {"id": batch_ids[j], "values": batch_vectors[j], "metadata": {"text": batch_texts[j]}}
            for j in range(len(batch_ids))
        ]
        kb_index.upsert(records)
    print("✅ All data indexed successfully.")

def retrieve_relevant_docs(query: str, top_k: int = 5):
    model = get_embedding_model()
    if kb_index is None:
        return []
    query_vec = model.encode([query])[0].tolist()
    results = kb_index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]


#                GEMINI RESPONSE

def generate_response(user_query: str, context_docs: List[str]) -> str:
    """Use Gemini 2.5 Flash for contextual response."""
    context = "\n\n".join(context_docs)
    prompt = f"""You are a helpful HR assistant. Use the context to answer the question clearly.

Context:
{context}

Question: {user_query}
Answer:"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini API Error: {e}")
        return "Sorry, I couldn’t process that request right now. Please try again later."


#                FASTAPI APP

app = FastAPI()
chat_memory = []

@app.on_event("startup")
def startup_event():
    print("🚀 Smart Assistant Booting...")
    print("⏭️ Skipping embedding on Render startup.")



# === Chat Request ===
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(chat: ChatRequest):
    global chat_memory
    try:
        memory_context = chat_memory[-3:]
        docs = retrieve_relevant_docs(chat.query)
        context = memory_context + docs
        answer = generate_response(chat.query, context)

        chat_memory.append(chat.query)
        chat_memory.append(answer)
        chat_memory = chat_memory[-6:]
        return {"answer": answer}
    except Exception as e:
        print("❌ Error in /chat:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Employee Model ===
class Employee(BaseModel):
    employee_id: str
    name: str
    email: str
    department: str
    joining_date: str
    role: str

@app.post("/add-employee")
def add_employee(employee: Employee):
    try:
        file_path = DATA_PATH

        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=["employee_id", "name", "email", "department", "joining_date", "role"])

        new_entry = {
            "employee_id": employee.employee_id,
            "name": employee.name,
            "email": employee.email,
            "department": employee.department,
            "joining_date": employee.joining_date,
            "role": employee.role,
        }

        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_excel(file_path, index=False)

        text = " | ".join(new_entry.values())
        embedding = embedding_model.encode([text])[0].tolist()
        kb_index.upsert([{"id": f"emp_{employee.employee_id}", "values": embedding, "metadata": {"text": text}}])

        return {"message": "✅ Employee added and indexed successfully."}
    except Exception as e:
        print("❌ Error adding employee:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


#                SIMPLE HTML FRONTEND

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
    <html>
        <head>
            <title>Smart Knowledge Assistant</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(to right, #ece9e6, #ffffff);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: start;
                    height: 100vh;
                    padding-top: 20px;
                    margin: 0;
                }
                h2, h3 { color: #333; margin: 10px 0; }
                form {
                    display: flex;
                    gap: 10px;
                    margin: 10px;
                    flex-wrap: wrap;
                    justify-content: center;
                }
                input[type="text"], input[type="email"], input[type="date"] {
                    padding: 8px;
                    font-size: 14px;
                    border: 2px solid #aaa;
                    border-radius: 6px;
                    width: 200px;
                }
                button {
                    padding: 10px 20px;
                    font-size: 14px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                }
                button:hover { background-color: #45a049; }
                pre {
                    background: #f4f4f4;
                    padding: 15px;
                    border-radius: 10px;
                    width: 80%;
                    max-width: 600px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }
            </style>
        </head>
        <body>
            <h2>Ask the Smart Knowledge Assistant</h2>
            <form onsubmit="submitChatForm(event)">
                <input type="text" id="query" placeholder="Type your question..." required />
                <button type="submit">Ask</button>
            </form>
            <pre id="response">Your answer will appear here...</pre>

            <h3>Add New Employee</h3>
            <form onsubmit="submitEmployeeForm(event)">
                <input type="text" id="employee_id" placeholder="Employee ID" required />
                <input type="text" id="name" placeholder="Name" required />
                <input type="email" id="email" placeholder="Email" required />
                <input type="text" id="department" placeholder="Department" required />
                <input type="date" id="joining_date" required />
                <input type="text" id="role" placeholder="Role" required />
                <button type="submit">Add Employee</button>
            </form>
            <pre id="emp_response">Employee status will appear here...</pre>

            <script>
                async function submitChatForm(event) {
                    event.preventDefault();
                    const query = document.getElementById('query').value;
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query})
                    });
                    const data = await response.json();
                    document.getElementById('response').innerText = data.answer || data.error || "No response.";
                }

                async function submitEmployeeForm(event) {
                    event.preventDefault();
                    const payload = {
                        employee_id: document.getElementById('employee_id').value,
                        name: document.getElementById('name').value,
                        email: document.getElementById('email').value,
                        department: document.getElementById('department').value,
                        joining_date: document.getElementById('joining_date').value,
                        role: document.getElementById('role').value
                    };
                    const response = await fetch('/add-employee', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });
                    const result = await response.json();
                    document.getElementById('emp_response').innerText = result.message || result.error;
                }
            </script>
        </body>
    </html>
    """


#                RUN COMMAND
#
# Run with: uvicorn app:app --reload
