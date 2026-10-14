from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Download NLTK stopwords if required
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Model files
MODEL_PATH = os.path.join("models", "model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"^.*?\([a-zA-Z\s]+\)\s*-\s*", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    cleaned_words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_words)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/samples_list")
def samples_list():
    import random
    fake_path = os.path.join("data", "fake.csv")
    true_path = os.path.join("data", "true.csv")
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        return jsonify({"error": "Dataset files not found."}), 404
    try:
        fake_df = pd.read_csv(fake_path).sample(25)
        true_df = pd.read_csv(true_path).sample(25)
        samples = []
        for _, row in fake_df.iterrows():
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            samples.append({"category": "FAKE", "title": title, "text": title + "\n\n" + text})
        for _, row in true_df.iterrows():
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            samples.append({"category": "REAL", "title": title, "text": title + "\n\n" + text})
        random.shuffle(samples)
        return jsonify({"samples": samples})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/live_news")
def live_news():
    import urllib.request
    import xml.etree.ElementTree as ET
    import html

    source = request.args.get("source", "bbc").lower()
    sources_map = {
        "bbc": ("BBC News", "https://feeds.bbci.co.uk/news/rss.xml"),
        "guardian": ("The Guardian", "https://www.theguardian.com/world/rss"),
        "npr": ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
        "aljazeera": ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        "timesofindia": ("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms")
    }

    if source not in sources_map:
        source = "bbc"

    source_name, feed_url = sources_map[source]

    try:
        req = urllib.request.Request(
            feed_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        res = urllib.request.urlopen(req, timeout=6)
        raw_xml = res.read()
        root = ET.fromstring(raw_xml)

        items = root.findall(".//item")
        articles = []
        for item in items[:30]:
            title_elem = item.find("title")
            desc_elem = item.find("description")

            title = title_elem.text.strip() if (title_elem is not None and title_elem.text) else ""
            desc = desc_elem.text.strip() if (desc_elem is not None and desc_elem.text) else ""

            # Unescape HTML entities & strip tags from description
            desc_clean = re.sub(r"<.*?>", "", html.unescape(desc)).strip()
            title_clean = html.unescape(title).strip()

            if title_clean:
                content = f"{title_clean}\n\n{desc_clean}" if desc_clean else title_clean
                articles.append({
                    "title": title_clean,
                    "text": content,
                    "source": source_name
                })

        return jsonify({"source": source_name, "articles": articles})
    except Exception as e:
        return jsonify({"error": f"Failed to fetch live feed: {str(e)}"}), 500


TRUSTED_NEWS_PROVIDERS = [
    "times of india", "the hindu", "indian express", "hindustan times",
    "ndtv", "bbc", "reuters", "guardian", "ap news", "associated press",
    "al jazeera", "cnn", "news18", "india today", "economic times",
    "livemint", "mint", "deccan herald", "tribune", "telegraph", "ani",
    "mid-day", "financial express", "business standard", "indiatimes"
]

DEBUNK_KEYWORDS = [
    "fake", "fact check", "fact-check", "hoax", "false", "debunk",
    "misleading", "rumour", "rumor", "busted", "fake alert"
]


def is_trusted_provider(source_name):
    s = source_name.lower()
    return any(p in s for p in TRUSTED_NEWS_PROVIDERS)


def verify_live_news_and_facts(text):
    """
    Cross-checks the submitted text with live news and fact-checking feeds.
    Detects if the story is confirmed by credible news media, or if
    it has been flagged as a hoax or fake news by fact-checkers.
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    words = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)]
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "have", "were", "been",
        "will", "would", "could", "should", "about", "after", "before", "their", "there",
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "than",
        "more", "some", "such", "into", "over", "under", "also", "just", "only", "very", "amid"
    }
    keywords = [w for w in words if w not in stop]
    if len(keywords) < 2:
        return None

    query = " ".join(keywords[:6])
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    try:
        res = urllib.request.urlopen(req, timeout=4)
        root = ET.fromstring(res.read())
        input_set = set(keywords)
        debunk_matches = []
        real_matches = []

        for item in root.findall(".//item")[:10]:
            title = (item.find("title").text or "").strip()
            source = item.find("source").text.strip() if (item.find("source") is not None and item.find("source").text) else "News Outlet"
            link = (item.find("link").text or "").strip()
            t_lower = title.lower()
            t_words = set(re.findall(r"[a-zA-Z]{3,}", t_lower))
            common = input_set.intersection(t_words)
            ratio = len(common) / len(input_set)

            is_debunk = any(dk in t_lower for dk in DEBUNK_KEYWORDS)

            match_info = {
                "title": title,
                "source": source,
                "link": link,
                "overlap": round(ratio * 100, 1)
            }

            if is_debunk and ratio >= 0.3:
                debunk_matches.append(match_info)
            elif ratio >= 0.55 and (is_trusted_provider(source) or ratio >= 0.70):
                real_matches.append(match_info)

        if debunk_matches:
            return {"status": "DEBUNKED_FAKE", "match": debunk_matches[0], "matches": debunk_matches}
        if real_matches:
            real_matches.sort(key=lambda x: x["overlap"], reverse=True)
            return {"status": "VERIFIED_REAL", "match": real_matches[0], "matches": real_matches}
    except Exception:
        pass

    return None


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
        text = data.get("text", "") if data else ""
    else:
        text = request.form.get("text", "")

    text = text.strip()

    if not text:
        if request.is_json:
            return jsonify({"error": "Please enter a news article."}), 400
        return render_template("result.html", error="Please enter a news article.")

    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        if request.is_json:
            return jsonify({"error": "The article does not contain enough usable text."}), 400
        return render_template("result.html", error="The article does not contain enough usable text.")

    # 1. Live cross-check against news & fact-checking records
    live_verification = verify_live_news_and_facts(text)

    # 2. Machine learning feature extraction & inference
    features = vectorizer.transform([cleaned_text])
    vocab_matches = int(features.nnz)
    word_count = len(text.split())

    fake_prob = 0.0
    real_prob = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        # index 0 = FAKE, index 1 = REAL
        fake_prob = round(float(probabilities[0]) * 100, 2)
        real_prob = round(float(probabilities[1]) * 100, 2)
        confidence = round(max(probabilities) * 100, 2)
    else:
        confidence = 0.0

    is_short_text = word_count < 20
    verified_info = None

    if live_verification and live_verification["status"] == "DEBUNKED_FAKE":
        result = "FAKE"
        confidence = 96.5
        fake_prob = 96.5
        real_prob = 3.5
        verified_info = {
            "type": "DEBUNKED",
            "title": live_verification["match"]["title"],
            "source": live_verification["match"]["source"],
            "link": live_verification["match"]["link"],
            "description": f"Debunked as false or misleading by fact-checkers ({live_verification['match']['source']})."
        }
    elif live_verification and live_verification["status"] == "VERIFIED_REAL":
        result = "REAL"
        top_match = live_verification["match"]
        confidence = max(86.0, min(98.5, top_match["overlap"]))
        real_prob = confidence
        fake_prob = round(100.0 - confidence, 2)
        verified_info = {
            "type": "VERIFIED",
            "title": top_match["title"],
            "source": top_match["source"],
            "link": top_match["link"],
            "description": f"Corroborated by verified reporting in {top_match['source']}."
        }
    else:
        # Fallback to ML classification
        margin = abs(fake_prob - real_prob)
        if confidence < 58.0 or margin < 16.0 or (is_short_text and confidence < 65.0):
            result = "UNCERTAIN"
        elif fake_prob > real_prob:
            result = "FAKE"
        else:
            result = "REAL"

    if request.is_json:
        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "word_count": word_count,
            "is_short_text": is_short_text,
            "vocab_matches": vocab_matches,
            "verified_info": verified_info
        })

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        fake_probability=fake_prob,
        real_probability=real_prob,
        news_text=text,
        word_count=word_count,
        is_short_text=is_short_text,
        vocab_matches=vocab_matches,
        verified_info=verified_info
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)