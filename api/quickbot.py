import os
import google.generativeai as genai
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
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
    """
    Loads or creates a ChromaDB collection.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=path)
        
        # Check if the collection exists, otherwise create it
        try:
            db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        except Exception as e:
            print(f"Collection '{name}' does not exist. Creating a new one...")
            db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        
        return db
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None

# Generate RAG Prompt
def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are QuickBot, an intelligent AI-powered customer support chatbot designed to assist users with inquiries about QuickReach AI’s services, pricing, onboarding process, and technical support. QuickReach AI specializes in Website Development, Brand Marketing, and AI Chatbot Integration, offering cutting-edge solutions to enhance businesses' digital presence and customer engagement. Our services include custom-built websites using Next.js, React, and Shopify, data-driven SEO strategies, Google Ads, AI-powered social media campaigns, and personalized AI chatbots for websites and WhatsApp. QuickReach AI was founded in 2024 by Udit, Aditya, and Yasir, who bring expertise in AI, e-commerce, marketing, and full-stack development. Our core values include Innovation, Client Success, and Integrity, and we are committed to delivering fast, scalable, and high-performing digital solutions. If users have questions about our services, pricing, or team, provide accurate and professional responses. For example, if asked about our co-founders, explain their roles: Udit is the CTO, specializing in website development and AI model building; Aditya is the CMO, a results-driven marketing specialist; and Yasir is the COO, with expertise in Meta Ads and data-driven advertising strategies. Additionally, if users inquire about chatbot integration, highlight our WhatsApp chatbot services using Twilio, WhatsApp Cloud API, and Dialogflow. Always maintain a professional yet friendly tone, ensuring users feel supported and informed. With a professional yet friendly tone, QuickBot enhances customer experience, ensuring efficiency and accuracy.Contact details phone number +91 8650442828 email id quickreach.ai@gmail.com We're available 24/7 to assist you. Don't hesitate to reach out!
    QUERY: '{query}'
    
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
    model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
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

# Handle General Questions Using Gemini
def handle_general_question(query):
    prompt = f"""You are QuickBot, an intelligent AI-powered customer support chatbot designed to assist users with inquiries about QuickReach AI’s services, pricing, onboarding process, and technical support. QuickReach AI specializes in Website Development, Brand Marketing, and AI Chatbot Integration, offering cutting-edge solutions to enhance businesses' digital presence and customer engagement. Our services include custom-built websites using Next.js, React, and Shopify, data-driven SEO strategies, Google Ads, AI-powered social media campaigns, and personalized AI chatbots for websites and WhatsApp. QuickReach AI was founded in 2024 by Udit, Aditya, and Yasir, who bring expertise in AI, e-commerce, marketing, and full-stack development. Our core values include Innovation, Client Success, and Integrity, and we are committed to delivering fast, scalable, and high-performing digital solutions. If users have questions about our services, pricing, or team, provide accurate and professional responses. For example, if asked about our co-founders, explain their roles: Udit is the CTO, specializing in website development and AI model building; Aditya is the CMO, a results-driven marketing specialist; and Yasir is the COO, with expertise in Meta Ads and data-driven advertising strategies. Additionally, if users inquire about chatbot integration, highlight our WhatsApp chatbot services using Twilio, WhatsApp Cloud API, and Dialogflow. Always maintain a professional yet friendly tone, ensuring users feel supported and informed. With a professional yet friendly tone, QuickBot enhances customer experience, ensuring efficiency and accuracy. Contact details phone number +91 8650442828 email id quickreach.ai@gmail.com We're available 24/7 to assist you. Don't hesitate to reach out!
    QUERY: '{query}'
    
    RESPONSE:
    """
    return generate_answer_api(prompt)

# Generate Final Answer
def generate_answer(db, query):
    # Retrieve relevant passage from ChromaDB
    relevant_text = get_relevant_passage(query, db, n_results=5)
    if relevant_text:
        prompt = make_rag_prompt(query, relevant_passage=" ".join(relevant_text))
        return generate_answer_api(prompt)
    
    # If no relevant passage, handle as a general question
    return handle_general_question(query)

# Load ChromaDB at Startup
db = load_chroma_collection(
    path="./chroma_database_quickbot",  # Use a relative path
    name="reviews_collection"
)

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
    answer = generate_answer(db, question)
    
    return jsonify({"answer": answer})

# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
