from sentence_transformers import SentenceTransformer 
import mysql.connector
import numpy as np
import faiss
import pickle

# Load multilingual embedding model
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

# Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="YJnov@06",
        database="farmer_schemes_db"
    )

# Fetch all schemes from the database
def fetch_schemes():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, scheme_name, eligibility FROM schemes")  
    schemes = cursor.fetchall()
    cursor.close()
    conn.close()
    return schemes

# Generate embeddings for all schemes
def generate_and_store_embeddings():
    schemes = fetch_schemes()
    
    if not schemes:
        print("No schemes found in the database!")
        return

    scheme_ids = []
    scheme_texts = []

    for scheme in schemes:
        text = f"{scheme['scheme_name']} - {scheme['eligibility']}"  # Fixed column names
        scheme_texts.append(text)
        scheme_ids.append(scheme["id"])

    # Generate embeddings
    embeddings = embedding_model.encode(scheme_texts)

    # Store embeddings using FAISS
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings, dtype=np.float32))

    # Save FAISS index and mapping
    faiss.write_index(faiss_index, "faiss_index.bin")

    with open("scheme_id_mapping.pkl", "wb") as f:
        pickle.dump(scheme_ids, f)

    print("Embeddings stored successfully!")

if __name__ == "__main__":
    generate_and_store_embeddings()
