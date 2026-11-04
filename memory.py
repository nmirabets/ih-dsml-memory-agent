import uuid
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime, timezone
import streamlit as st

# Initialize Pinecone for vector database
pc = Pinecone(st.secrets["PINECONE_API_KEY"])
# Initialize the vector database index
index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])
# Initialize OpenAI for embeddings 
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_embeddings(string_to_embed):
    response = client.embeddings.create(
        input=string_to_embed,
        model=st.secrets["EMBEDDING_MODEL"],
        dimensions=int(st.secrets["EMBEDDING_DIMENSIONS"]),
    )
    return response.data[0].embedding

def save_memory(memories):
    for memory in memories:
        # Step 1: Embed the memory
        vector = get_embeddings(memory)
        # Step 2: Build the vector document to be stored
        user_id = "1234"
        current_time = datetime.now(tz=timezone.utc)

        documents = [
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {
                    "user_id": user_id,
                    "timestamp": str(current_time),
                    "payload": memory,
                },
            }
        ]
        # Step 3: Store the vector document in the vector database
        index.upsert(
            vectors=documents,
            namespace=st.secrets["PINECONE_NAMESPACE"]
        )
        st.caption(f"Memory updated: {memory}")
    return f"{len(memories)} memories saved successfully"

def load_memories(prompt):
    user_id = "1234"
    top_k = 4
    vector = get_embeddings(prompt)
    response = index.query(
        vector=vector,
        filter={
            "user_id": {"$eq": user_id},
        },
        namespace=st.secrets["PINECONE_NAMESPACE"],
        include_metadata=True,
        top_k=top_k,
    )
    return [m["metadata"]["payload"] for m in response.get("matches", [])]