import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# --- Configuration Streamlit ---
st.set_page_config(page_title="Nutribot 🍃", layout="centered", page_icon="🍊")

# --- Style personnalisé ---
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
    }
    h1 {
        text-align: center;
        color: #2E8B57;
        font-size: 2.8rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #FFA726;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        border: none;
    }
    .stTextArea label {
        color: #2E8B57;
        font-weight: bold;
    }
    .stTextArea textarea {
        border: 2px solid #2E8B57;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logo ---
st.image("nutribot_logo.png", width=130)  # Ajuste le chemin selon ton projet

# --- Titre principal ---
st.markdown("<h1>Nutribot 🍃</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Votre assistant nutrition basé sur vos documents scientifiques 📚</div>", unsafe_allow_html=True)

#  Chargement du modèle d'embeddings HuggingFace
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"} 
)


vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = Ollama(
    model="tinyllama",
    base_url="http://localhost:11434"
)

# Création du prompt personnalisé
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Vous êtes un expert en nutrition. Répondez à la question ci-dessous de manière claire, professionnelle et fiable. "
        "Utilisez en priorité les informations fournies, mais vous pouvez les compléter si nécessaire avec vos connaissances.\n\n"
        "Informations disponibles :\n{context}\n\n"
        "Question : {question}\n\n"
        "Réponse :"
    )
)

qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

rag_chain = RetrievalQAWithSourcesChain(
    combine_documents_chain=qa_chain,
    retriever=retriever,
    return_source_documents=True
)

# --- Interface utilisateur ---
st.markdown("---")

user_input = st.text_area("🍽 Posez votre question nutritionnelle :", placeholder="Ex. : Quels aliments pour améliorer le fer ?", height=100)

if st.button("🔍 Poser la question"):
    if user_input.strip():
        with st.spinner("🔄 Analyse de votre question..."):
            try:
                result = rag_chain({"question": user_input})
                response = result["answer"]
                sources = result["source_documents"]

                st.markdown("### 🧠 Réponse de Nutribot")
                st.success(response)

                st.markdown("### 📖 Sources utilisées")
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get('source', 'Inconnu')
                    st.expander(f"📘 Source {i} : {source_name}").write(doc.page_content[:600] + "...")

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {e}")
    else:
        st.warning("⚠️ Veuillez entrer une question.")

