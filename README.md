# RagBase - Private Chat with Your Documents

> Completely local RAG with chat UI

<a href="https://www.mlexpert.io/bootcamp" target="_blank">
  <img src="https://raw.githubusercontent.com/curiousily/ragbase/master/.github/ui.png">
</a>

## Demo

Check out the [RagBase on Streamlit Cloud](https://ragbase.streamlit.app/). Runs with Groq API.

## Installation

Clone the repo:

```sh
git clone git@github.com:curiousily/ragbase.git
cd ragbase
```

Install the dependencies (requires Poetry):

```sh
poetry install
```

Fetch your LLM (gemma2:9b by default):

```sh
ollama pull gemma2:9b
```

Run the Ollama server

```sh
ollama serve
```

Start RagBase (Streamlit UI — optional):

```sh
poetry run streamlit run app.py
```

## CLI Usage (no frontend required)

RagBase ships with a `main.py` CLI that lets you ingest documents and ask
questions without starting the Streamlit app.

### Ingest documents

Scan a directory (recursively) for supported files (`.pdf`, `.png`, `.jpg`,
`.jpeg`) and index them into the local Qdrant vector store:

```sh
python main.py ingest ./path/to/docs
```

The command exits with a non-zero status and a descriptive message if the
directory does not exist or no supported files are found.

### Ask a question

Query the existing vector store and print the answer to stdout:

```sh
python main.py ask "What is the main topic of the documents?"
```

Output format:

```
=== ANSWER ===
<answer text>
```

> **Note:** The `ask` command requires that you have already run `ingest` at
> least once (or used the Streamlit UI to upload files) so that the vector
> store exists.

## Image Ingestion (OCR)

RagBase can extract text from image files (PNG, JPG, JPEG) via OCR and index
that text alongside your PDFs so you can ask questions about scanned documents
and screenshots.

### Install Tesseract OCR

Tesseract is a system dependency that must be installed separately:

| Platform | Command |
|----------|---------|
| **macOS** | `brew install tesseract` |
| **Ubuntu / Debian** | `sudo apt-get install tesseract-ocr` |
| **Windows** | Download the installer from the [UB-Mannheim wiki](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH` |

The Python wrappers (`pytesseract` and `Pillow`) are already included in the
project dependencies and will be installed by `poetry install` / `pip install`.

### Usage

**Via CLI (recommended, no frontend required):**

```sh
python main.py ingest ./path/to/images_and_pdfs
python main.py ask "What does the scanned document say?"
```

**Via Streamlit UI:**

1. Start the app (`poetry run streamlit run app.py`).
2. In the file-upload widget you can now select **PDF, PNG, JPG, or JPEG** files.
3. After uploading, a brief OCR preview (first 200 characters) is shown for
   each image.
4. Chat with your documents as usual — OCR text is indexed in the same vector
   store as PDF content.

### Limitations & Troubleshooting

- **Quality depends on image clarity.** Blurry or low-contrast images will
  produce poor OCR results. A grayscale + contrast-boost preprocessing step is
  applied automatically to improve accuracy on most scanned documents.
- **Languages.** By default Tesseract uses English. For other languages install
  the appropriate language pack (e.g. `tesseract-ocr-chi-sim` for Simplified
  Chinese on Ubuntu) and pass `lang=` to `pytesseract.image_to_string`.
- **Tesseract not found error.** If you see *"Tesseract OCR is not installed or
  not found on PATH"* follow the installation table above and restart the app.

## Architecture

<a href="https://www.mlexpert.io/bootcamp" target="_blank">
  <img src="https://raw.githubusercontent.com/curiousily/ragbase/master/.github/architecture.png">
</a>

### Ingestor

Extracts text from PDF documents and images (via OCR) and creates chunks (using semantic and character splitter) that are stored in a vector database.

### Retriever

Given a query, searches for similar documents, reranks the result and applies LLM chain filter before returning the response.

### QA Chain

Combines the LLM with the retriever to answer a given user question

## Tech Stack

- [Ollama](https://ollama.com/) - run local LLM
- [Groq API](https://groq.com/) - fast inference for mutliple LLMs
- [LangChain](https://www.langchain.com/) - build LLM-powered apps
- [Qdrant](https://qdrant.tech/) - vector search/database
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - fast reranking
- [FastEmbed](https://qdrant.github.io/fastembed/) - lightweight and fast embedding generation
- [Streamlit](https://streamlit.io/) - build UI for data apps
- [PDFium](https://pdfium.googlesource.com/pdfium/) - PDF processing and text extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) + [pytesseract](https://github.com/madmaze/pytesseract) - image text extraction

## Add Groq API Key (Optional)

You can also use the Groq API to replace the local LLM, for that you'll need a `.env` file with Groq API key:

```sh
GROQ_API_KEY=YOUR API KEY
```