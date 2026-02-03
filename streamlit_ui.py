import streamlit as st
from gtts import gTTS
import io
import csv
import os
from backend.pdf_parser import extract_text_from_pdf, chunk_text
from backend.embeddings import embed_chunks, embed_query
from backend.retrieval import FAISSRetriever
from backend.llm_integration import ask_llm

def text_to_speech(answer_text):
    # Generate audio in memory (not saved to disk)
    tts = gTTS(text=answer_text, lang="en")
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)  # rewind to start
    return audio_bytes

st.set_page_config(page_title="StudyMate", layout="wide")

st.title("📘 StudyMate - AI PDF Q&A")

# Initialize Q&A history in session
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    st.success("✅ PDFs uploaded and processed!")

    embeddings = embed_chunks(all_chunks)
    retriever = FAISSRetriever(embeddings)

    question = st.text_input("Ask a question:")
    if st.button("Submit") and question:
        query_emb = embed_query(question)
        indices = retriever.retrieve(query_emb, k=3)
        context = [all_chunks[i] for i in indices]

        answer = ask_llm(question, context)

        # --- Show Answer ---
        st.markdown(f"### Answer:\n{answer}")
        with st.expander("📖 Read More (Source Paragraphs)"):
            for i, c in enumerate(context, 1):
                st.markdown(f"**Source {i}:**\n{c}")

        # --- Audio playback ---
        audio_bytes = text_to_speech(answer)
        st.audio(audio_bytes, format="audio/mp3")

        # --- Save to session history ---
        st.session_state.qa_history.append({"question": question, "answer": answer})

        # --- Append to logs/qa_history.txt ---
        os.makedirs("logs", exist_ok=True)
        with open("logs/qa_history.txt", "a", encoding="utf-8") as f:
            f.write(f"Q: {question}\nA: {answer}\n{'-'*40}\n")

# --- Download Buttons ---
if st.session_state.qa_history:
    # TXT format
    txt_output = io.StringIO()
    for i, qa in enumerate(st.session_state.qa_history, 1):
        txt_output.write(f"Q{i}: {qa['question']}\n")
        txt_output.write(f"A{i}: {qa['answer']}\n")
        txt_output.write("-" * 40 + "\n")

    st.download_button(
        label="📥 Download Q&A History (TXT)",
        data=txt_output.getvalue(),
        file_name="qa_history.txt",
        mime="text/plain"
    )

    # CSV format
    csv_output = io.StringIO()
    writer = csv.writer(csv_output)
    writer.writerow(["Question", "Answer"])  # header
    for qa in st.session_state.qa_history:
        writer.writerow([qa["question"], qa["answer"]])

    st.download_button(
        label="📥 Download Q&A History (CSV)",
        data=csv_output.getvalue(),
        file_name="qa_history.csv",
        mime="text/csv"
    )