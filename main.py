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
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
import logging

# Configuration (overridable via environment for Cloud Run)
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-Coder-3B-Instruct")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "all-MiniLM-L6-v2")
DB_PATH = os.getenv("DB_PATH", "database.db")
DATA_DIR = os.getenv("DATA_DIR", "../data")
# os.cpu_count() can report host cores on Cloud Run; pin threads to allocated vCPUs
torch.set_num_threads(int(os.getenv("TORCH_THREADS", os.cpu_count() or 1)))

# Configure Logging to write to pipeline.log and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", mode="a"),  # Logs saved here
        logging.StreamHandler(),  # Keeps terminal logs active
    ],
)
logger = logging.getLogger("Text2SQL_RAG")

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
    print(f"Loading {MODEL_ID} on CPU...")
    models["generator"] = pipeline(
        "text-generation",
        model=MODEL_ID,
        device=-1,
        dtype=torch.bfloat16,
        model_kwargs={"low_cpu_mem_usage": True, 
                      "use_cache": True  # Enables key-value caching during generation
                     },     
    )

    print("Loading Embedding Model...")
    models["embedding_fn"] = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_ID, device="cpu"
        )
    )
    # 1a. Initialize the SQLite database with CSV data
    # Rebuild from CSVs only when they are present (local dev). In the
    # container the prebuilt database.db is baked into the image.
    csv_files = [os.path.join(DATA_DIR, f) for f in ("PUBSJOURNALS.csv", "AUTHORS.csv")]
    if all(os.path.exists(f) for f in csv_files):
        try:
            for f in csv_files:
                process_file(f)
        except Exception as e:
            print(f"Error during database initialization: {e}")
    else:
        print(f"CSV files not found in {DATA_DIR}; using existing {DB_PATH}")
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"Database file {DB_PATH} not found")


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
    # Read-only: LLM-generated SQL must never modify the data
    models["conn"] = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
    )

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
# AGENT 0: QUERY REWRITER (Resolves follow-ups using Chat History)
# =====================================================================
def rewrite_query_with_history(current_question: str, history: list) -> str:
    # If this is the first question (no history), don't waste time rewriting it
    if not history or len(history) == 0:
        return current_question

    # Format the history into a readable string for the LLM
    history_text = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a query rewriting assistant. Rewrite the user's current question into a "
                "standalone, fully-contextualized query using the conversation history.\n"
                "Do not answer the question. Output ONLY the rewritten question string."
            )
        },
        # FEW-SHOT EXAMPLE
        {
            "role": "user",
            "content": (
                "History:\nUser: How many publications are from Berkeley in 2021?\n"
                "Assistant: There are 63 publications from Berkeley in 2021.\n\n"
                "Current Question: What about Stanford?"
            )
        },
        {
            "role": "assistant",
            "content": "How many publications are from Stanford in 2021?"
        },
        # REAL INPUT
        {
            "role": "user",
            "content": f"History:\n{history_text}\n\nCurrent Question: {current_question}"
        }
    ]

    prompt = model_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    rewritten = model_pipeline(
        prompt, 
        max_new_tokens=50, 
        max_length=None, 
        do_sample=False
    )[0]["generated_text"][len(prompt):].strip()
    
    return rewritten
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
    
    return model_pipeline(prompt, max_new_tokens=150, max_length=None, do_sample=False)[0]["generated_text"][len(prompt):].strip()

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
                    "3. Do NOT add conversational explanation text.\n"
                    "4. CRITICAL CONSTRAINT:Use ONLY column names explicitly listed in the provided Schema. NEVER invent columns.\n"
                    "5.CRITICAL: If a requested column does not exist in the schema, select only the closest matching available column.\n"
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
    
    raw_output = model_pipeline(prompt, max_new_tokens=150, max_length=None, do_sample=False)[0]["generated_text"][len(prompt):]
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
        max_length=None,
        do_sample=False
    )[0]["generated_text"][len(prompt):]
    
    return raw_response.strip()

# Pass lifespan to FastAPI initialization
app = FastAPI(title="Text2SQL RAG API", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[Message] = []

@app.get("/health")
def health_check():
  return {"status": "ok", "message": "Text2SQL RAG service is running"}


@app.post("/query")
def process_query(req: QueryRequest):
    logger.info("==================================================")
    logger.info(f"[USER REQUEST] Question: {req.question}")
    logger.info(f"[USER REQUEST] History: {req.history}")

    # 1. AGENT 0: Rewrite the query if there is history
    standalone_question = rewrite_query_with_history(req.question, req.history)
    print(f"\n[Original] {req.question}")
    print(f"[Rewritten] {standalone_question}\n")
    logger.info(f"[AGENT 0 - REWRITER] Rewritten Question: {standalone_question}")

    # 2. Retrieve Schema Context using the STANDALONE QUESTION
    schema_results = collection.query(query_texts=[standalone_question], n_results=2)
    retrieved_schemas = "\n\n".join(schema_results["documents"][0])
    logger.info(f"[AGENT 0 - SCHEMA RETRIEVAL] Retrieved Schemas: {retrieved_schemas}")

    # 3. AGENT 1: Generate Plan
    plan = analyze_schema_plan(standalone_question, retrieved_schemas)
    logger.info(f"[AGENT 1 - PLAN GENERATION] Plan: {plan}")
    
    # 4. AGENT 2: Generate SQL & Execute with Loop
    attempts = 0
    max_retries = 3
    history = None
    last_error = ""
    sql = ""

    while attempts < max_retries:
        attempts += 1
        # Pass standalone_question here
        sql, history, raw_output = generate_sql_from_plan(
            standalone_question, retrieved_schemas, plan, history
        )
        logger.info(
        f"[AGENT 2 - SQL GENERATOR] Attempt {attempts}/{max_retries} Generated"
        f" SQL:\n{sql}"
        )
        
        try:
            cursor = db_conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()

            logger.info(f"[TOOL CALL - SQLite] Executed Query Successfully.")
            logger.info(f"[TOOL CALL - SQLite] Raw Query Output: {results}")

            # EXTRACT COLUMN HEADERS FROM CURSOR DESCRIPTION
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            logger.info(f"[TOOL CALL - SQLite] Column Headers: {columns}")

            # Guardrail:(Fixes Result Fabrications)
            if not results or results == [()] or results == [(None,)]:
                return {
                     "answer": "No matching records were found in the database for your request."
                        }
            
            logger.info(f"[GUARDRAIL] Query returned empty set. Returning default.")
            logger.info("==================================================\n")
            # Only invoke Agent 3 if valid data actually exists
            # 5. AGENT 3: Generate Natural Language Answer
            # Pass standalone_question here as well
            # If the SQL returns tabular data for a report, skip Agent 3 generation
            if len(results) > 1:
                final_answer = f"Generated report returning {len(results)} rows:"
            else:
                final_answer = generate_natural_response(standalone_question, results)
            logger.info(f"[AGENT 3 - RESPONSE GENERATOR] Final Natural Answer:\n{final_answer}")
            logger.info("==================================================\n")
            return {
                "answer": final_answer,
                "columns": columns,
                "data": results,
                "generated_sql": sql
            }

        except Exception as e:
            last_error = str(e)
            logger.error(f"[TOOL CALL - SQLite] Execution Failed (Attempt" 
                         f" {attempts}): {last_error}"
                        )
            if history is not None:
                history.append({"role": "assistant", "content": raw_output})
                history.append({
                    "role": "user",
                    "content": f"Your SQL failed with SQLite error: {last_error}. Fix the query."
                })
    logger.error(f"[PIPELINE ERROR] Max retries reached. Request failed.")
    logger.info("==================================================\n")
    raise HTTPException(
        status_code=500, detail={"error": "Failed", "backend_error": last_error}
    )