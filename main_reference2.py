import os
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

# Global variables for app state
model_pipeline = None
collection = None
db_conn = None
# Global variables for persistent memory loading
models = {}


def extract_clean_sql(llm_output: str) -> str:
  code_block_match = re.search(
      r"```(?:sql)?\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE
  )
  if code_block_match:
    return code_block_match.group(1).strip()
  select_match = re.search(
      r"(SELECT.*?;)", llm_output, re.DOTALL | re.IGNORECASE
  )
  if select_match:
    return select_match.group(1).strip()
  return llm_output.strip()


# Define the Lifespan Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, collection, db_conn

    # 1. Load heavy models ONCE on server startup
    print("Loading 1.5B LLM on CPU...")
    models["generator"] = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-Coder-3B-Instruct",
        device=-1,
        dtype=torch.bfloat16,
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
    # 1. Index Schemas into ChromaDB
    schema_documents = [
        (
            "Table: AUTHORS\nColumns: alsid (INT), lastname (TEXT), firstname"
            " (TEXT), institution (TEXT), dbpubid (INT)\nDescription: Contains"
            " publication authors and institutions. dbpubid = publication ID;"
            " alsid = person ID."
        ),
        (
            "Table: PUBSJOURNALS\nColumns: dbpubid (INT), artchapthesttle (TEXT),"
            " pubyear (INT), refereed (BOOLEAN), beamline (TEXT), journalcode"
            " (INT), journaltitle (TEXT), journalimpactfactor (TEXT),"
            " doehighimpact (BOOL), builtauthorlist (TEXT), publisting"
            " (TEXT)\nDescription: Contains publications, year (pubyear), title"
            " (artchapthesttle), and impact metrics."
        ),
    ]
    models["collection"].upsert(
        documents=schema_documents, ids=["table_authors", "table_pubsjournals"]
    )

    # 4. Connect to SQLite
    models["conn"] = sqlite3.connect("database.db", check_same_thread=False)

    # Expose loaded resources via module-level globals used by request handlers
    model_pipeline = models["generator"]
    collection = models["collection"]
    db_conn = models["conn"]

    print("Pipeline ready for requests!")
    yield

    print("Cleaning up resources...closing database connection...freeing memory...done")
    # --- SHUTDOWN LOGIC ---
    if db_conn:
      db_conn.close()


# =====================================================================
# AGENT 1: SCHEMA ANALYZER (Identifies required Tables, Columns, JOINs)
# =====================================================================
def analyze_schema_plan(question: str, schemas: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a database query planner. Analyze the schema and user question to plan a SQL query.\n"
                "Output strictly in this format:\n"
                "TARGET_TABLES: <tables needed>\n"
                "JOIN_CONDITIONS: <join keys or NONE>\n"
                "AGGREGATION_AND_COLUMNS: <columns/functions to select>\n"
                "FILTERS: <WHERE clause filters>"
            )
        },
        # FEW-SHOT 1: Multi-table JOIN with Count Distinct
        {
            "role": "user",
            "content": (
                "Schema:\n"
                "Table: AUTHORS (alsid INT, lastname TEXT, firstname TEXT, institution TEXT, dbpubid INT)\n"
                "Table: PUBSJOURNALS (dbpubid INT, artchapthesttle TEXT, pubyear INT, refereed BOOL, journaltitle TEXT, doehighimpact BOOL)\n\n"
                "Question: What is the distinct total number of publications in year 2021 from institution: University of California Berkeley?"
            )
        },
        {
            "role": "assistant",
            "content": (
                "TARGET_TABLES: PUBSJOURNALS, AUTHORS\n"
                "JOIN_CONDITIONS: PUBSJOURNALS.dbpubid = AUTHORS.dbpubid\n"
                "AGGREGATION_AND_COLUMNS: COUNT(DISTINCT PUBSJOURNALS.dbpubid)\n"
                "FILTERS: PUBSJOURNALS.pubyear = 2021 AND AUTHORS.institution LIKE '%University of California Berkeley%'"
            )
        },
        # FEW-SHOT 2: Multi-table JOIN with Boolean Filter
        {
            "role": "user",
            "content": (
                "Schema:\n"
                "Table: AUTHORS (alsid INT, lastname TEXT, firstname TEXT, institution TEXT, dbpubid INT)\n"
                "Table: PUBSJOURNALS (dbpubid INT, artchapthesttle TEXT, pubyear INT, refereed BOOL, journaltitle TEXT, doehighimpact BOOL)\n\n"
                "Question: Show titles of DOE high impact publications co-authored by researchers from Stanford."
            )
        },
        {
            "role": "assistant",
            "content": (
                "TARGET_TABLES: PUBSJOURNALS, AUTHORS\n"
                "JOIN_CONDITIONS: PUBSJOURNALS.dbpubid = AUTHORS.dbpubid\n"
                "AGGREGATION_AND_COLUMNS: DISTINCT PUBSJOURNALS.artchapthesttle\n"
                "FILTERS: PUBSJOURNALS.doehighimpact = 1 AND AUTHORS.institution LIKE '%Stanford%'"
            )
        },
        # REAL RUNTIME INPUT
        {
            "role": "user",
            "content": f"Schema:\n{schemas}\n\nQuestion: {question}"
        }
    ]

    prompt = model_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    return model_pipeline(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"][len(prompt):].strip()

# =====================================================================
# AGENT 2: SQL CODER (Generates SQL using the Plan + Few-Shot Examples)
# =====================================================================
def generate_sql_from_plan(question: str, schemas: str, plan: str, history: list = None) -> tuple[str, list, str]:
    if history is None:
        history = [
            {
                "role": "system",
                "content": (
                    "You are a SQLite code generator. Write a valid query using the plan and schema.\n"
                    "Rules:\n"
                    "1. Output ONLY valid SQL inside ```sql ... ``` code blocks.\n"
                    "2. Use aliases `p` for PUBSJOURNALS and `a` for AUTHORS.\n"
                    "3. Do NOT add conversational explanation text."
                )
            },
            # FEW-SHOT 1: Mapping exact column names (artchapthesttle, pubyear, dbpubid)
            {
                "role": "user",
                "content": (
                    "Schema:\n"
                    "Table: AUTHORS (alsid INT, lastname TEXT, firstname TEXT, institution TEXT, dbpubid INT)\n"
                    "Table: PUBSJOURNALS (dbpubid INT, artchapthesttle TEXT, pubyear INT, refereed BOOL, journaltitle TEXT, doehighimpact BOOL)\n\n"
                    "Plan:\n"
                    "TARGET_TABLES: PUBSJOURNALS, AUTHORS\n"
                    "JOIN_CONDITIONS: PUBSJOURNALS.dbpubid = AUTHORS.dbpubid\n"
                    "AGGREGATION_AND_COLUMNS: COUNT(DISTINCT PUBSJOURNALS.dbpubid)\n"
                    "FILTERS: PUBSJOURNALS.pubyear = 2021 AND AUTHORS.institution LIKE '%University of California Berkeley%'\n\n"
                    "Question: What is the distinct total number of publications in year 2021 from institution: University of California Berkeley?"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "```sql\n"
                    "SELECT COUNT(DISTINCT p.dbpubid) AS total_publications\n"
                    "FROM PUBSJOURNALS p\n"
                    "JOIN AUTHORS a ON p.dbpubid = a.dbpubid\n"
                    "WHERE p.pubyear = 2021\n"
                    "  AND a.institution LIKE '%University of California Berkeley%';\n"
                    "```"
                )
            },
            # REAL RUNTIME INPUT
            {
                "role": "user",
                "content": f"Schema:\n{schemas}\n\nPlan:\n{plan}\n\nQuestion: {question}"
            }
        ]

    prompt = model_pipeline.tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    
    raw_output = model_pipeline(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"][len(prompt):]
    clean_sql = extract_clean_sql(raw_output)
    return clean_sql, history, raw_output
# =====================================================================
# AGENT 3: DATA ANALYST (Converts raw SQL results into natural language)
# =====================================================================
def generate_natural_response(question: str, sql_result: list) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful data assistant. Given a user's question and the raw SQL query result, "
                "formulate a concise, natural language answer. Do not mention SQL, databases, or tables."
            )
        },
        {
            "role": "user",
            "content": f"Question: {question}\nData Result: {sql_result}\n\nProvide the final answer:"
        }
    ]

    prompt = model_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    raw_response = model_pipeline(
        prompt, 
        max_new_tokens=100, 
        do_sample=False
    )[0]["generated_text"][len(prompt):]
    
    return raw_response.strip()

# Pass lifespan to FastAPI initialization
app = FastAPI(title="Text2SQL RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
  question: str


@app.get("/health")
def health_check():
  return {"status": "ok", "message": "Text2SQL RAG service is running"}


@app.post("/query")
def process_query(req: QueryRequest):
    # 1. Retrieve Schema Context
    schema_results = collection.query(query_texts=[req.question], n_results=2)
    retrieved_schemas = "\n\n".join(schema_results["documents"][0])

    # 2. AGENT 1: Generate Plan
    plan = analyze_schema_plan(req.question, retrieved_schemas)
    
    # LOGGING: Print Plan to terminal instead of returning it to user
    print(f"\n[{req.question}] --- AGENT 1 PLAN ---\n{plan}\n")

    # 3. AGENT 2: Generate SQL & Execute with Loop
    attempts = 0
    max_retries = 3
    history = None
    last_error = ""
    sql = ""

    while attempts < max_retries:
        attempts += 1
        sql, history, raw_output = generate_sql_from_plan(
            req.question, retrieved_schemas, plan, history
        )
        
        # LOGGING: Print SQL attempt to terminal
        print(f"[{req.question}] --- AGENT 2 SQL (Attempt {attempts}) ---\n{sql}\n")

        try:
            cursor = db_conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()

            # LOGGING: Print raw result to terminal
            print(f"[{req.question}] --- SQL RAW RESULT ---\n{results}\n")

            # 4. AGENT 3: Generate Natural Language Answer
            final_answer = generate_natural_response(req.question, results)
            
            # LOGGING: Print final answer
            print(f"[{req.question}] --- AGENT 3 FINAL ANSWER ---\n{final_answer}\n")

            # RETURN ONLY THE CLEAN ANSWER TO THE USER
            return {
                "answer": final_answer
            }

        except Exception as e:
            last_error = str(e)
            print(f"[{req.question}] --- SQLite Error ---\n{last_error}\n")
            
            if history is not None:
                history.append({"role": "assistant", "content": raw_output})
                history.append({
                    "role": "user",
                    "content": (
                        f"Your SQL failed with SQLite error: {last_error}. Please fix"
                        " the query and output ONLY the corrected SQL inside ```sql"
                        " ... ``` blocks."
                    ),
                })

    raise HTTPException(
        status_code=500,
        detail={"error": "Failed to process query", "backend_error": last_error},
    )