# SailPoint Database Chatbot

A sophisticated database chatbot that uses AI to generate SQL queries from natural language questions. Built with Streamlit, Google Gemini AI, and MySQL.

## Features

- **Natural Language to SQL**: Convert plain English questions into SQL queries using Google Gemini AI
- **Database Schema Analysis**: Automatically analyzes MySQL database schemas
- **Security Validation**: Comprehensive SQL injection prevention and query validation
- **Audit Logging**: Complete audit trail of all queries and user actions
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Real-time Results**: Execute queries and display results in a beautiful web interface

## Prerequisites

- Python 3.9+
- MySQL database
- Google Gemini API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Bayyana-kiran/SailPoint.git
cd SailPoint
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file with:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=your_database_name
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
```

## ChromaDB Setup

The application uses ChromaDB for semantic context and query enrichment. You can run ChromaDB in two modes:

### Option 1: ChromaDB Server (Recommended)

1. Install ChromaDB server:

```bash
pip install chromadb
```

2. Start the ChromaDB server on localhost:8000:

```bash
./start_chromadb.sh
```

Or manually:

```bash
chroma run --host 0.0.0.0 --port 8000
```

### Option 2: Local ChromaDB (Fallback)

If the server is not available, the application will automatically fall back to using a local ChromaDB instance with persistent storage.

### Configuration

ChromaDB settings can be configured in the `.env` file:

```env
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Ask questions about your database in natural language!

## Example Questions

- "Show me all users from the HR department"
- "What are the top 10 products by sales?"
- "How many orders were placed last month?"
- "Which employees have access to sensitive applications?"
- "Show me audit events from the last 7 days"

## Architecture

- **Frontend**: Streamlit web application
- **AI Engine**: Google Gemini for SQL generation
- **Database Layer**: SQLAlchemy with connection pooling
- **Security**: Custom SQL validator with injection prevention
- **Audit**: Comprehensive logging system
- **Rate Limiting**: Token bucket algorithm

## Security Features

- SQL injection prevention
- Query validation against database schema
- Rate limiting per user and globally
- Audit logging of all activities
- Input sanitization
- Restricted operations (read-only by default)

## Development

Run tests:

```bash
pytest tests/ -v
```

Format code:

```bash
black .
```

Lint code:

```bash
flake8 .
```

## Configuration

The application uses several configuration files:

- `config/database.py`: Database connection settings
- `config/security.py`: Security policies and restrictions
- `config/gemini.py`: AI model configuration

## License

This project is part of the SailPoint IdentityIQ ecosystem.
