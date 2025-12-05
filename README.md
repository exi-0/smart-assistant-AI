# 🤖 Smart Assistant AI  
An AI-powered Knowledge Assistant built with **FastAPI**, **Gemini 2.5 Flash**, **Sentence Transformers**, and **Pinecone Vector Database**.  
This assistant can ingest PDFs, CSVs, Excel files, and provide intelligent HR/employee support answers using RAG (Retrieval Augmented Generation).

---

## 🚀 Features

### ✅ **Retrieval-Augmented Generation (RAG)**
- Indexes **PDF**, **CSV**, and **Excel** documents  
- Embeds text using **SentenceTransformer MiniLM-L6-v2**  
- Stores embeddings in **Pinecone**

### ✅ **AI-Powered Answers (Gemini 2.5 Flash)**
- Uses contextual retrieval + LLM reasoning  
- Responds like an HR assistant  
- Understands multi-turn chat with small memory buffer

### ✅ **FastAPI Backend**
- `/chat` endpoint → Ask questions  
- `/add-employee` endpoint → Add new employee and auto-index  
- Runs fast and supports real frontend integration

### ✅ **Built-in Minimal HTML UI**
- Search bar for chat  
- Form to add employees  
- Clean frontend served directly by FastAPI

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Backend | FastAPI |
| LLM | Gemini 2.5 Flash |
| Vector DB | Pinecone |
| Embeddings | all-MiniLM-L6-v2 |
| Models | PyTorch |
| File Handling | PyPDF2, Pandas, Excel |
| Deployment | Render / Railway |

---

## 📁 Project Structure

smart-assistant-AI/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── data/
│ ├── complaint_knowledge.csv
│ ├── employee_data.xlsx
│ ├── All_Employee_Report.pdf


---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/exi-0/smart-assistant-AI.git
cd smart-assistant-AI
2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Create .env File

Add your keys:

PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key

▶️ Running the App Locally

Start the FastAPI server:

uvicorn app:app --reload


Open in browser:

http://127.0.0.1:8000/


You will see:

Chat assistant interface

Add employee form

Answer box

☁️ Deploying on Render
Settings:

Build Command

pip install -r requirements.txt


Start Command

uvicorn app:app --host 0.0.0.0 --port 8000


Add environment variables:

PINECONE_API_KEY=xxxx
GEMINI_API_KEY=xxxx


Then click Deploy.

🧪 API Routes
🔹 POST /chat

Ask questions to the assistant.

Request:

{
  "query": "Show all employees who joined after 2021"
}


Response:

{
  "answer": "Here are the employees..."
}

🔹 POST /add-employee

Request:

{
  "employee_id": "101",
  "name": "John Doe",
  "email": "john@example.com",
  "department": "IT",
  "joining_date": "2024-01-12",
  "role": "Developer"
}


Response:

Employee added and indexed successfully.

📌 Notes

Do not upload .env to GitHub

Indexing runs at startup — ensure your data/ folder exists

Pinecone index is automatically created if missing

⭐ Future Improvements

JWT Auth for admin panel

Streamlit / React UI

Multi-language support

Employee analytics dashboard

📜 License

MIT License © 2025 Exi-0

❤️ Support

If you like this project, consider giving a ⭐ on GitHub!

