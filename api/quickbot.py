import os
import google.generativeai as genai
import chromadb
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up Gemini API Key
os.environ["GEMINI_API_KEY"] = "AIzaSyC--4S4iuAK_z8mTAgKsZiLHq5kMS1D_K4"

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()

# Load CSV File
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.set_index('Question')['Answer'].to_dict()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}

# Define Custom Embedding Function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key is missing! Set GEMINI_API_KEY as an environment variable.")
        
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"

        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# Load ChromaDB Collection
def load_chroma_collection(path, name):
    try:
        chroma_client = chromadb.PersistentClient(path=path)
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        return db
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None

# Generate RAG Prompt
def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are an intelligent customer support chatbot named QuickBot, designed to assist users with inquiries about QuickReach AI’s services, pricing, onboarding process, and technical support. QuickBot operates in two modes: General Support Mode (handling FAQs about services, pricing, onboarding, and troubleshooting) and Lead Generation Mode (engaging clients, explaining value propositions, collecting inquiries, and suggesting relevant services). With a professional yet friendly tone, QuickBot enhances customer experience, ensuring efficiency and accuracy.
    
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)

    return prompt

# Generate Answer Using Gemini API
def generate_answer_api(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key is missing! Set GEMINI_API_KEY as an environment variable.")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    try:
        answer = model.generate_content(prompt)
        return answer.text if answer else "Sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate a response."

# Preprocess Query
def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    tokens = word_tokenize(query)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

# Retrieve Relevant Passage from ChromaDB
def get_relevant_passage(query, db, n_results=5):
    try:
        processed_query = preprocess_query(query)
        expanded_query = expand_query_with_synonyms(processed_query)
        result = db.query(query_texts=[expanded_query], n_results=n_results)
        passage = result['documents'][0] if 'documents' in result and result['documents'] else ""
        return passage
    except Exception as e:
        print(f"Error retrieving passage: {e}")
        return ""

# Expand Query with Synonyms
def expand_query_with_synonyms(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    
    domain_synonyms = {
        "quickreach ai": ["quickreach", "quickreachai", "quick reach ai"],
        "services": ["offerings", "solutions", "products"],
        "co-founders": ["founders", "creators", "team"],
    }
    for key, values in domain_synonyms.items():
        if key in query:
            synonyms.update(values)
    
    expanded_query = query + " " + " ".join(synonyms)
    return expanded_query

# Check CSV for Exact or Synonym-Based Match
def check_csv_for_answer(query, csv_data):
    processed_query = preprocess_query(query)
    
    if processed_query in csv_data:
        return csv_data[processed_query]
    
    expanded_query = expand_query_with_synonyms(processed_query)
    for word in expanded_query.split():
        if word in csv_data:
            return csv_data[word]
    
    return None

# Handle General Questions Using Gemini
def handle_general_question(query):
    prompt = f"""You are an intelligent customer support chatbot named QuickBot, designed to assist users with inquiries about QuickReach AI’s services, pricing, onboarding process, and technical support. QuickBot operates in two modes: General Support Mode (handling FAQs about services, pricing, onboarding, and troubleshooting) and Lead Generation Mode (engaging clients, explaining value propositions, collecting inquiries, and suggesting relevant services). With a professional yet friendly tone, QuickBot enhances customer experience, ensuring efficiency and accuracy. Respond to the following query in a conversational tone:
    
    QUERY: '{query}'
    
    RESPONSE:
    """
    return generate_answer_api(prompt)

# Generate Final Answer
def generate_answer(db, query, csv_data):
    csv_answer = check_csv_for_answer(query, csv_data)
    if csv_answer:
        return csv_answer
    
    relevant_text = get_relevant_passage(query, db, n_results=5)
    if relevant_text:
        prompt = make_rag_prompt(query, relevant_passage=" ".join(relevant_text))
        return generate_answer_api(prompt)
    
    return handle_general_question(query)

# Load ChromaDB and CSV at Startup
db = load_chroma_collection(
    path="C:/Users/LENOVO/OneDrive/Desktop/QuickReach/api/chroma_database_quickbot",
    name="reviews_collection"
)
csv_data = load_csv("C:/Users/LENOVO/OneDrive/Desktop/QuickReach/api/quickbot_faq_updated.csv")

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Define API Endpoint
@app.route("/chatbot", methods=["POST"])
def get_answer():
    if db is None:
        return jsonify({"error": "Database not loaded. Please check ChromaDB setup."}), 500
    
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' parameter"}), 400

    question = data["question"]
    answer = generate_answer(db, question, csv_data)
    
    return jsonify({"answer": answer})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)