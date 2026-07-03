# PageSage: Persistent Document Q&A Web App

PageSage is a lightweight, local Document Q&A application that allows users to upload plain text files and ask questions directly about their contents. The application chunks and embeds the document locally using a SentenceTransformer model, stores these vectors inside a persistent Chroma database, and leverages Gemini 2.5 Flash to generate precise, grounded answers. Every answer generated is accompanied by the exact numbered source passages used from the document to ensure factual correctness and accountability.

## Features

- Upload and process plain text (`.txt`) documents
- Automatic document chunking and embedding
- Persistent vector storage using ChromaDB
- Semantic retrieval with SentenceTransformers
- Grounded answer generation using Gemini 2.5 Flash
- Source passage citations for every answer
- Conversation history
- Document preview
- One-click document reset and re-indexing
- Protection against empty queries


## Tech Stack

- Python
- Streamlit
- SentenceTransformers
- ChromaDB
- Google Gemini 2.5 Flash
- python-dotenv

## Interface Overview
* **File Uploader:** Drag-and-drop area supporting `.txt` files.
* **Status Flags & Warning Banner:** Displays chunk indices on success, and gives a warning if the document is too short (< 5 chunks).
* **Reset Trigger:** A "Clear and re-index" button to wipe the current document's Chroma collection and re-ingest.
* **Document Previewer:** A collapsible expander showing the first 500 characters of the document.
* **Conversation History:** A sequential list of previously asked questions, answers, and their corresponding source passages.
* **Interactive Query Form:** An input box and "Ask" button that are disabled until a document is ingested, with protection against empty inputs.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/absheerakk/pagesage.git
cd pagesage
```
### 2. Configure Environment Variables
Create a file named `.env` in the root directory:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Web Application
```bash
streamlit run app.py
```
This will start a local server and open the web application automatically in your browser (usually at `http://localhost:8501`).
## Known Limitations
* **Supported File Types:** Currently only supports plain text (`.txt`) files.
* **Language Support:** Optimized primarily for English documents and queries.
* **Context Windows:** Relies on retrieving the top 3 most relevant sentences/chunks; extremely dense or sprawling documents might miss broader contextual inferences.
* **Local Storage:** The persistent vector database is stored locally inside the project directory under `chroma_db/` and is excluded from source control.