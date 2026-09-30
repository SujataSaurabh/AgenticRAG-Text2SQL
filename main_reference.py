import re
import sqlite3
from contextlib import asynccontextmanager
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import pipeline
from datapreprocess import process_file

# # Ensure the database is populated with initial CSV data
# process_file("../data/PUBSJOURNALS.csv")
# process_file("../data/AUTHORS.csv")

# Global variables for persistent memory loading
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load heavy models ONCE on server startup
    print("Loading 1.5B LLM on CPU...")
    models["generator"] = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        device=-1,
        dtype=torch.float32,
        model_kwargs={"low_cpu_mem_usage": True},
    )

    print("Loading Embedding Model...")
    models["embedding_fn"] = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cpu"
        )
    )
    # 1a. Initialize the SQLite database with CSV data
    try:
        process_file("../data/PUBSJOURNALS.csv")
        process_file("../data/AUTHORS.csv")
    except Exception as e:
        print(f"Error during database initialization: {e}")    


    # 2. Setup Vector Database
    chroma_client = chromadb.Client()
    models["collection"] = chroma_client.get_or_create_collection(
        name="db_schema", embedding_function=models["embedding_fn"]
    )

    # 3. Index Schemas
    schema_documents = [
     
    "Table: AUTHORS\nColumns: alsid (INT), lastname (TEXT), firstname (TEXT), institution (TEXT), dbpubid (INT)\nDescription:  The table contains the authors of publications, first name, last name and their institutions. dbpubid: unique identifier of publications; alsid: unique identifier of people",
    "Table: PUBSJOURNALS\nColumns: dbpubid (INT), artchapthesttle (TEXT), pubyear (INT), refereed (BOOLEAN), beamline (TEXT), journalcode (INT), journaltitle (TEXT), journalimpactfactor (TEXT), doehighimpact (BOOL), builtauthorlist (TEXT), publisting (TEXT), \nDescription: Table contains the publications (dbpubid), published year (pubyear), title (artchapthesttle), journal (journal code is the unique journal identifier), journal title (journaltitle), journal impact factor (journalimpactfactor), if it is considered a doe high imaoct journal (doehighimpact) or not."

    ]
    models["collection"].upsert(
        documents=schema_documents, ids=["table_authors", "table_pubsjournals"]
    )

    # 4. Connect to SQLite
    models["conn"] = sqlite3.connect("database.db", check_same_thread=False)

    print("Pipeline ready for requests!")
    yield
    print("Cleaning up resources...closing database connection...freeing memory...done")
    # Cleanup on server shutdown
    models["conn"].close()


app = FastAPI(title="Text-to-SQL RAG API", lifespan=lifespan)

# @app.on_event("startup")
# def startup_db_init():
#     try:
#         process_file("../data/PUBSJOURNALS.csv")
#         process_file("../data/AUTHORS.csv")
#     except Exception as e:
#         print(f"Error during database initialization: {e}")

class QueryRequest(BaseModel):
    question: str


def extract_clean_sql(llm_output: str) -> str:
    code_block = re.search(
        r"```(?:sql)?\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE
    )
    if code_block:
        return code_block.group(1).strip()
    select_match = re.search(
        r"(SELECT.*?;)", llm_output, re.DOTALL | re.IGNORECASE
    )
    return select_match.group(1).strip() if select_match else llm_output.strip()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def process_query(request: QueryRequest):
    user_question = request.question
    if not user_question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # A. Schema Retrieval
    results = models["collection"].query(
        query_texts=[user_question], n_results=2
    )
    schemas = "\n\n".join(results["documents"][0])

    # B. Generate SQL
    sql_prompt = models["generator"].tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert SQLite generator. Write a valid SQLite"
                    " query based ONLY on the schema. Output ONLY raw SQL"
                    " ending with a semicolon."
                ),
            },
            {
                "role": "user",
                "content": f"Schema:\n{schemas}\n\nQuestion: {user_question}",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    raw_sql = models["generator"](
        sql_prompt,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=models["generator"].tokenizer.eos_token_id,
    )[0]["generated_text"][len(sql_prompt) :]

    clean_sql = extract_clean_sql(raw_sql)

    # C. Execute SQL
    try:
        cursor = models["conn"].cursor()
        cursor.execute(clean_sql)
        query_result = cursor.fetchall()
    except Exception as e:
        return {"error": f"SQL Execution Failed: {str(e)}", "sql": clean_sql}

    # D. Synthesize Answer
    ans_prompt = models["generator"].tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "Formulate a concise natural language response based on"
                    " query results."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {user_question}\nSQL: {clean_sql}\nData:"
                    f" {query_result}\n\n"
                    "Answer the question clearly:"
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    final_answer = models["generator"](
        ans_prompt,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=models["generator"].tokenizer.eos_token_id,
    )[0]["generated_text"][len(ans_prompt) :]

    return {
        "question": user_question,
        "generated_sql": clean_sql,
        "raw_data": query_result,
        "answer": final_answer.strip(),
    }