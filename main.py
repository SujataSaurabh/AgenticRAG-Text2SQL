import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

import db
import llm
import schema_catalog
from load_data import load_sqlite

# Configuration (overridable via environment for Cloud Run)
DATA_DIR = os.getenv("DATA_DIR", "../data")

# Full schema description for the planner and SQL coder, from the catalog.
# Add retrieval (pick relevant tables per question) once the catalog grows.
SCHEMA_TEXT = schema_catalog.schema_text()
LIKE = db.LIKE_OP  # LIKE for SQLite, ILIKE for PostgreSQL (case-insensitive)

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
    # 1. Initialize the LLM backend once on server startup
    llm.init()

    # 2. Local dev convenience: rebuild the SQLite file when the CSVs are
    # present. In the container the prebuilt database.db is bundled.
    if db.DB_BACKEND == "sqlite":
        csvs = [os.path.join(DATA_DIR, t.csv_file) for t in schema_catalog.TABLES]
        if all(os.path.exists(f) for f in csvs):
            load_sqlite(DATA_DIR, db.DB_PATH)
        else:
            print(f"CSV files not found in {DATA_DIR}; using existing {db.DB_PATH}")
    db.check_connection()  # fail fast if the database is unreachable

    print("Pipeline ready for requests!")
    yield
    db.close()
    print("Shutting down.")

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

    rewritten = llm.generate(messages, max_new_tokens=50)
    
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
                f"FILTERS: PUBSJOURNALS.pubyear = 2021 AND AUTHORS.institution {LIKE} '%University of California Berkeley%'"
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
                f"FILTERS: PUBSJOURNALS.doehighimpact = TRUE AND AUTHORS.institution {LIKE} '%Stanford%'"
            )
        },
        # REAL RUNTIME INPUT
        {
            "role": "user",
            "content": f"Schema:\n{schemas}\n\nQuestion: {question}"
        }
    ]

    return llm.generate(messages, max_new_tokens=150)

# =====================================================================
# AGENT 2: SQL CODER (Generates SQL using the Plan + Few-Shot Examples)
# =====================================================================
def generate_sql_from_plan(question: str, schemas: str, plan: str, history: list = None) -> tuple[str, list, str]:
    if history is None:
        history = [
            {
                "role": "system",
                "content": (
                    f"You are a {db.DIALECT_NAME} code generator. Write one valid, read-only SELECT query using the plan and schema.\n"
                    "Rules:\n"
                    "1. Output ONLY valid SQL inside ```sql ... ``` code blocks.\n"
                    "2. Use aliases `p` for PUBSJOURNALS and `a` for AUTHORS.\n"
                    "3. Do NOT add conversational explanation text.\n"
                    "4. CRITICAL CONSTRAINT:Use ONLY column names explicitly listed in the provided Schema. NEVER invent columns.\n"
                    "5.CRITICAL: If a requested column does not exist in the schema, select only the closest matching available column.\n"
                    f"6. Use {LIKE} for text matching and TRUE/FALSE for boolean columns.\n"
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
                    f"FILTERS: PUBSJOURNALS.pubyear = 2021 AND AUTHORS.institution {LIKE} '%University of California Berkeley%'\n\n"
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
                    f"  AND a.institution {LIKE} '%University of California Berkeley%';\n"
                    "```"
                )
            },
            # REAL RUNTIME INPUT
            {
                "role": "user",
                "content": f"Schema:\n{schemas}\n\nPlan:\n{plan}\n\nQuestion: {question}"
            }
        ]

    raw_output = llm.generate(history, max_new_tokens=150)
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

    raw_response = llm.generate(messages, max_new_tokens=100)
    
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

    # 2. Schema context: the full schema (see SCHEMA_DOCUMENTS)
    retrieved_schemas = SCHEMA_TEXT

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
            columns, results = db.run_query(sql)

            logger.info(f"[TOOL CALL - DB] Executed Query Successfully.")
            logger.info(f"[TOOL CALL - DB] Raw Query Output: {results}")

            # EXTRACT COLUMN HEADERS FROM CURSOR DESCRIPTION
            logger.info(f"[TOOL CALL - DB] Column Headers: {columns}")

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
            logger.error(f"[TOOL CALL - DB] Execution Failed (Attempt" 
                         f" {attempts}): {last_error}"
                        )
            if history is not None:
                history.append({"role": "assistant", "content": raw_output})
                history.append({
                    "role": "user",
                    "content": f"Your SQL failed with {db.DIALECT_NAME} error: {last_error}. Fix the query."
                })
    logger.error(f"[PIPELINE ERROR] Max retries reached. Request failed.")
    logger.info("==================================================\n")
    raise HTTPException(
        status_code=500, detail={"error": "Failed", "backend_error": last_error}
    )