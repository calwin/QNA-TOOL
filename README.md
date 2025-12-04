# Q&A Intelligence Tool

A premium Question & Answer tool powered by Azure OpenAI, capable of analyzing multiple document formats including PDF, Excel, XML, DBC, ARXML, CDD, A2L, and Python files.

## Prerequisites

- Python 3.9+
- Node.js 16+
- Azure OpenAI API Key and Endpoint

## Setup

### 1. Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure Environment Variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and fill in your Azure OpenAI details.

5. Run the server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will start at `http://localhost:8000`.

### 2. Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies (if not already done):
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   The frontend will start at `http://localhost:5173`.

## Usage

1. Open the frontend URL in your browser.
2. Upload your documents using the panel on the left.
3. Wait for the processing to complete.
4. Ask questions in the chat window on the right.

## Supported File Formats

- **PDF** - Text extraction from PDF documents
- **Excel** (.xlsx, .xls) - Spreadsheet data analysis
- **XML** - Generic XML file parsing
- **DBC** - CAN database files with message and signal definitions
- **ARXML** - AUTOSAR XML files
- **CDD** - Calibration data definition files
- **A2L** - ASAP2 calibration files
- **Python** (.py) - Python code with function and class extraction

## Features

- 📄 Multi-format document support
- 🤖 AI-powered question answering with Azure OpenAI
- 🎨 Beautiful syntax-highlighted code generation
- 📊 Intelligent source citations
- 🚗 Automotive-specific file format support (DBC, ARXML, CDD, A2L)
- 🔍 Advanced RAG (Retrieval Augmented Generation) with relevance filtering
- 💻 Python code analysis and documentation extraction
