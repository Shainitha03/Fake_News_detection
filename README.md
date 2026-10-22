# Fake News Detection & Fact Verification System

An advanced Natural Language Processing (NLP) and Machine Learning project that evaluates news articles and claims as **Real News** or **Fake News** using machine learning stylometry, explainable AI token attribution, and multi-stage grounded fact verification with authoritative evidence citations.

## Key Features
- **Multi-Stage Grounded Fact Verification**:
  - **Google Fact Check Tools API**: Dynamic integration with Google's Fact Check Tools API to cross-reference claims against international certified fact-checking organizations (Snopes, PolitiFact, Full Fact, BOOM, AFP, Reuters Fact Check, Factly).
  - **Authoritative Fact & Viral Hoax Registry**: Evaluates viral claims with official statements (NASA, WHO, PIB Fact Check, RBI, UNESCO).
  - **Government Gazettes & Statutory Days**: Validates national days, events, and official statutory records.
  - **Live Encyclopedic Knowledge (Wikipedia REST API)**: Dynamically verifies subject entities, geographical consistency, and debunks pseudoscientific assertions.
  - **Political Leadership & Office Verification**: Validates constitutional roles, prime ministers, presidents, and corporate leadership against official records.
  - **Real-Time News Wire Cross-Checking**: Filters social media noise and verifies claims against certified fact-checkers and trusted news providers.
- **Explainable AI (XAI)**:
  - Token-level feature attribution showing exactly which linguistic markers and vocabulary patterns influenced the classification.
- **Authoritative Evidence Presentation**:
  - Provides direct excerpts from official reference documents, gazettes, and fact-checking agencies alongside every verdict.
  - Clean, percentage-free modern dark/light mode interface.

## Project Structure
- `data/`: Dataset storage for training and test samples.
- `models/`: Serialized TF-IDF vectorizer (`vectorizer.pkl`) and trained classification model (`model.pkl`).
- `src/`: Text preprocessing, feature extraction, and model training pipelines.
- `templates/`: Modern Jinja2 HTML templates (`index.html`, `result.html`).
- `static/`: Modern styling (`css/style.css`) and client-side application logic (`js/main.js`).
- `app.py`: Flask web application and factual verification engine.
- `.env`: Local environment configuration (keeps private API keys secure and ignored by Git).
- `.env.example`: Template for environment variables.

## Quick Start (Run Locally)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key (Optional but Recommended)**:
   Copy `.env.example` to `.env` and set your Google Fact Check Tools API key:
   ```bash
   # In .env:
   FACT_CHECK_API_KEY=your_api_key_here
   ```

3. **Start the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://127.0.0.1:5000` to analyze articles and verify factual claims.

