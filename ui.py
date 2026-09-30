import streamlit as st
import requests
import os
import pandas as pd

# --- CONFIGURATION ---
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8080/query")
# Set on Cloud Run: the API service URL, used as the ID-token audience so the
# UI's service account can call the private API. Unset locally.
API_AUDIENCE = os.getenv("API_AUDIENCE")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))


def auth_headers() -> dict:
    """Return an Authorization header with a Google ID token when deployed."""
    if not API_AUDIENCE:
        return {}
    import google.auth.transport.requests
    import google.oauth2.id_token

    token = google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), API_AUDIENCE
    )
    return {"Authorization": f"Bearer {token}"}

st.set_page_config(page_title="ALS Database AI Assistant", page_icon="🤖")
 # --- CUSTOM CSS FOR LARGER FONTS ---
st.markdown(
    """
    <style>
    /* Target all paragraphs inside chat messages (greeting + full history) */
    [data-testid="stChatMessageContent"] p {
        font-size: 1.55rem !important;  /* Adjust size as needed (e.g., 1.2rem, 1.4rem) */
        line-height: 1.6 !important;
        font-weight: 400;
    }

    /* Keep the chat input box matching the larger size */
    [data-testid="stChatInput"] textarea {
        font-size: 1.55rem !important;
    }
    /* Bolder and slightly larger section headers */
    h1 {
        font-size: 1.75rem !important;
    }

    /* General body text fallback */
    p, div {
        font-size: 1.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 ALS Database AI Assistant")
st.markdown("Ask questions about Authors, Institutions, and Publications!")

# --- INITIALIZE CHAT HISTORY ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask a question or request a report (e.g., *'Generate a report of all authors from Stanford'*).",
            "columns": None,
            "data": None,
        }
    ]

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this message contained table/report data, render the table & download button
        if message.get("data") and message.get("columns"):
            df = pd.DataFrame(message["data"], columns=message["columns"])
            st.dataframe(df, use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Report as CSV",
                data=csv_data,
                file_name="generated_report.csv",
                mime="text/csv",
                key=f"dl_{hash(message['content'])}"
            )
# Handle user input
if user_input := st.chat_input("E.g., Select alsid, firstname, lastname from AUTHORS"):
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input, "columns": None, "data": None})

    with st.chat_message("assistant"):
        with st.spinner("Processing query and generating report..."):
            try:
                chat_history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages[1:-1]
                ]

                response = requests.post(
                    FASTAPI_URL,
                    json={"question": user_input, "history": chat_history},
                    headers=auth_headers(),
                    timeout=REQUEST_TIMEOUT
                )

                if response.status_code == 200:
                    payload = response.json()
                    answer = payload.get("answer", "")
                    columns = payload.get("columns", [])
                    data = payload.get("data", [])

                    st.markdown(answer)

                    # If data rows were returned, render interactive DataFrame & Download button
                    if data and columns:
                        df = pd.DataFrame(data, columns=columns)
                        st.dataframe(df, width='stretch')

                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Report as CSV",
                            data=csv_data,
                            file_name="generated_report.csv",
                            mime="text/csv"
                        )

                    # Save complete response to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "columns": columns,
                        "data": data
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"⚠️ Backend connection failed. Check that the API at {FASTAPI_URL} is running.")
            except requests.exceptions.Timeout:
                st.error("⚠️ The request to the backend timed out. Please try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
# --- HANDLE USER INPUT ---
# if user_input := st.chat_input("E.g., How many publications are from Berkeley in 2021?"):
    
#     # 1. Display User Message
#     with st.chat_message("user"):
#         st.markdown(user_input)
    
#     # 2. Add to Session State (Memory)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # 3. Call FastAPI Backend
#     with st.chat_message("assistant"):
#         with st.spinner("Analyzing schema and querying database..."):
#             try:
#                 # Grab all chat history EXCEPT the message the user just typed (that's the current query)
#                 # Ignore the very first hardcoded greeting to save tokens
#                 chat_history = st.session_state.messages[1:-1] 

#                 # Send both the current question AND the history
#                 payload = {
#                     "question": user_input,
#                     "history": chat_history
#                 }

#                 response = requests.post(
#                     FASTAPI_URL,
#                     json=payload,
#                     timeout=60 
#                 )
                
#                 if response.status_code == 200:
#                     answer = response.json().get("answer", "I couldn't find an answer.")
#                     st.markdown(answer)
#                     st.session_state.messages.append({"role": "assistant", "content": answer})
#                 else:
#                     st.error(f"Backend Error: {response.status_code} - {response.text}")
            
#             except requests.exceptions.ConnectionError:
#                 st.error("⚠️ Could not connect to the backend. Is FastAPI running on port 8080?")
#             except requests.exceptions.Timeout:
#                 st.error("⚠️ The request to the backend timed out. Please try again.")
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {str(e)}")