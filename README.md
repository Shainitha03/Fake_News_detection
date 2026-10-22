# Fake News Detection & Fact Verification System

An advanced Natural Language Processing (NLP) and Machine Learning project that evaluates news articles and claims as **Real News** or **Fake News** using machine learning stylometry, explainable AI token attribution, and multi-stage grounded fact verification with authoritative evidence citations.

## Key Features
- **Multi-Stage Grounded Fact Verification**:
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

## Quick Start (Run Locally)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://127.0.0.1:5000` to analyze articles and verify factual claims.

