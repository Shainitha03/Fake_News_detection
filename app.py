from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup

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


PUBLISHER_WATERMARKS = [
    r'\b(?:reuters|ap|associated press|afp|bbc|cnn|nbc|cbs|fox news|the guardian|bloomberg)\b',
    r'\b(?:featured image|image via|photo by|photo credit|via getty|getty images|shutterstock)\b',
    r'\b(?:video below|watch the video|watch below|watch)\b',
    r'\b(?:21st century wire|infowars|breitbart|daily caller|zerohedge|the gateway pundit)\b',
    r'\b(?:click here|share this story|read more|source)\b'
]


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\b[a-zA-Z\s]{2,20}\s*\([a-zA-Z\s]+\)\s*[-–—]\s*", " ", text)

    for watermark in PUBLISHER_WATERMARKS:
        text = re.sub(watermark, " ", text, flags=re.IGNORECASE)

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

DISALLOWED_DOMAINS = [
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "youtube.com", "threads.net", "pinterest.com", "reddit.com", "quora.com",
    "linkedin.com", "tumblr.com"
]

MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}

OFFICIAL_INDIAN_RECORDS = {
    "independence": {
        "keywords": ["independence", "day"],
        "official_date": "15 august",
        "title": "National Days of India: Independence Day",
        "source": "Official Portal of Government of India (india.gov.in / PIB)",
        "link": "https://www.india.gov.in",
        "evidence": "Official Government of India records and gazettes designate 15 August as Independence Day, celebrated annually to commemorate freedom from British colonial rule on 15 August 1947."
    },
    "republic": {
        "keywords": ["republic", "day"],
        "official_date": "26 january",
        "title": "National Days of India: Republic Day",
        "source": "Official Portal of Government of India (india.gov.in)",
        "link": "https://www.india.gov.in",
        "evidence": "Republic Day is officially celebrated on 26 January to mark the adoption of the Constitution of India in 1950."
    },
    "gandhi_jayanti": {
        "keywords": ["gandhi", "jayanti"],
        "official_date": "2 october",
        "title": "National Holidays of India: Gandhi Jayanti",
        "source": "Official Portal of Government of India (india.gov.in)",
        "link": "https://www.india.gov.in",
        "evidence": "Gandhi Jayanti is celebrated on 2 October annually as a national holiday commemorating the birth anniversary of Mahatma Gandhi."
    }
}


def is_trusted_provider(source_name):
    s = source_name.lower()
    return any(p in s for p in TRUSTED_NEWS_PROVIDERS)


def is_disallowed_source(source_name, link=""):
    combined = (source_name + " " + link).lower()
    return any(domain in combined for domain in DISALLOWED_DOMAINS)


def normalize_text(text):
    # Normalize ordinals like 17th -> 17, 1st -> 1, 2nd -> 2, 3rd -> 3
    text = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)\b', r'\1', text.lower())
    return text


def extract_dates_and_numbers(text):
    text_norm = normalize_text(text)
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_norm)
    tokens = text_clean.split()
    dates = set()
    numbers = set()
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) <= 4:
            numbers.add(tok)
        if tok in MONTH_NAMES:
            if i > 0 and tokens[i-1].isdigit():
                day = int(tokens[i-1])
                if 1 <= day <= 31:
                    dates.add(f"{day} {tok}")
            if i + 1 < len(tokens) and tokens[i+1].isdigit():
                day = int(tokens[i+1])
                if 1 <= day <= 31:
                    dates.add(f"{day} {tok}")
    return dates, numbers


def verify_factual_claim_pipeline(text):
    """
    Performs comprehensive multi-stage factual verification:
    1. Checks Official Government References for national days, events, and statutory dates.
    2. Queries authoritative encyclopedic records (Wikipedia reference summaries).
    3. Cross-checks reputable news media feeds while rejecting social media (e.g. Facebook, Twitter)
       and requiring strict semantic & date/entity agreement.
    4. Evaluates conflicting evidence to produce:
       - 'Supported'
       - 'Contradicted'
       - 'Mixed or Misleading'
       - 'Unable to Verify'
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET
    import json

    norm = normalize_text(text)
    claim_dates, claim_numbers = extract_dates_and_numbers(norm)
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", norm)]
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "have", "were", "been",
        "will", "would", "could", "should", "about", "after", "before", "their", "there",
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "than",
        "more", "some", "such", "into", "over", "under", "also", "just", "only", "very",
        "celebrated", "celebrate", "celebrates", "national", "annual", "annually", "observed",
        "held", "year", "every", "holiday"
    }
    subject_words = [w for w in words if w not in MONTH_NAMES and w not in stop]
    base_query = " ".join(subject_words)

    # ----------------------------------------------------
    # STAGE 1: Check Official Government Sources (India & Global)
    # ----------------------------------------------------
    for key, rec in OFFICIAL_INDIAN_RECORDS.items():
        if all(kw in words for kw in rec["keywords"]):
            if claim_dates:
                claim_str = ", ".join(sorted(claim_dates))
                if rec["official_date"] in claim_dates:
                    return {
                        "status": "Supported",
                        "title": rec["title"],
                        "source": rec["source"],
                        "link": rec["link"],
                        "explanation": f"The claim is Supported by official government reference: {rec['title']} is celebrated on {rec['official_date']}.",
                        "evidence": rec["evidence"]
                    }
                else:
                    return {
                        "status": "Contradicted",
                        "title": rec["title"],
                        "source": rec["source"],
                        "link": rec["link"],
                        "explanation": f"Claim Contradicted by Evidence: The statement claims date '{claim_str}', but official government records ({rec['source']}) establish that this national day is celebrated on {rec['official_date']}.",
                        "evidence": rec["evidence"]
                    }

    # ----------------------------------------------------
    # STAGE 2: Authoritative Encyclopedic Verification (Wikipedia API)
    # ----------------------------------------------------
    if base_query.strip():
        search_queries = []
        if "india" in words or "indian" in words or not any(c in words for c in ["usa", "us", "pakistan", "israel", "france", "mexico", "canada"]):
            search_queries.append(base_query + " India")
        search_queries.append(base_query)

        for q in search_queries:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(q)}&srlimit=4&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "FakeNewsDetector/1.0 (academic-fact-check)"})
            try:
                with urllib.request.urlopen(req, timeout=3) as res:
                    results = json.loads(res.read().decode("utf-8")).get("query", {}).get("search", [])
                for item in results:
                    title = item.get("title", "")
                    if any(skip in title.lower() for skip in ["film", "album", "song", "novel", "soundtrack", "resurgence", "school"]):
                        continue
                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    sum_req = urllib.request.Request(sum_url, headers={"User-Agent": "FakeNewsDetector/1.0 (academic-fact-check)"})
                    with urllib.request.urlopen(sum_req, timeout=3) as s_res:
                        s_data = json.loads(s_res.read().decode("utf-8"))
                        extract = s_data.get("extract", "")
                        ref_dates, ref_numbers = extract_dates_and_numbers(extract)
                        ref_link = s_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title}")

                        if claim_dates and ref_dates:
                            claim_str = ", ".join(sorted(claim_dates))
                            ref_str = ", ".join(sorted(ref_dates))
                            if claim_dates.intersection(ref_dates):
                                return {
                                    "status": "Supported",
                                    "title": title,
                                    "source": f"Wikipedia Reference: {title}",
                                    "link": ref_link,
                                    "explanation": f"The claim is Supported by authoritative reference: {title} is commemorated on {ref_str}.",
                                    "evidence": extract[:240] + "..." if len(extract) > 240 else extract
                                }
                            else:
                                return {
                                    "status": "Contradicted",
                                    "title": title,
                                    "source": f"Wikipedia Reference: {title}",
                                    "link": ref_link,
                                    "explanation": f"Claim Contradicted by Evidence: The statement claims '{claim_str}', but the authoritative reference for '{title}' records '{ref_str}'.",
                                    "evidence": extract[:240] + "..." if len(extract) > 240 else extract
                                }
            except Exception:
                pass

    # ----------------------------------------------------
    # STAGE 3: Credible News & Fact-Check Feed Cross-Check
    # (Strict Relevance Check & Social Media Filtering)
    # ----------------------------------------------------
    keywords = [w for w in words if w not in stop]
    if len(keywords) >= 2:
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
            conflicting_matches = []

            for item in root.findall(".//item")[:10]:
                title = (item.find("title").text or "").strip()
                source = item.find("source").text.strip() if (item.find("source") is not None and item.find("source").text) else "News Outlet"
                link = (item.find("link").text or "").strip()

                # Requirement 5: Do not use Facebook or social media posts as credible evidence
                if is_disallowed_source(source, link):
                    continue

                t_lower = normalize_text(title)
                t_words = set(re.findall(r"[a-zA-Z]{3,}", t_lower))
                common = input_set.intersection(t_words)
                ratio = len(common) / len(input_set) if len(input_set) > 0 else 0

                is_debunk = any(dk in t_lower for dk in DEBUNK_KEYWORDS)
                item_dates, item_numbers = extract_dates_and_numbers(t_lower)
                has_date_conflict = bool(claim_dates and item_dates and not claim_dates.intersection(item_dates))

                match_info = {
                    "title": title,
                    "source": source,
                    "link": link,
                    "overlap": round(ratio * 100, 1)
                }

                if is_debunk and ratio >= 0.35:
                    debunk_matches.append(match_info)
                elif has_date_conflict and ratio >= 0.50:
                    conflicting_matches.append(match_info)
                elif not has_date_conflict and ratio >= 0.70 and (is_trusted_provider(source) or ratio >= 0.80):
                    real_matches.append(match_info)

            # Check for Debunked Hoaxes
            if debunk_matches:
                top = debunk_matches[0]
                return {
                    "status": "Contradicted",
                    "title": top["title"],
                    "source": top["source"],
                    "link": top["link"],
                    "explanation": f"Claim Contradicted by Evidence: Debunked as false or misleading by verified fact-checkers ({top['source']}).",
                    "evidence": top["title"]
                }

            # Check for Conflicting / Misleading Evidence
            if conflicting_matches and real_matches:
                top = conflicting_matches[0]
                return {
                    "status": "Mixed or Misleading",
                    "title": top["title"],
                    "source": top["source"],
                    "link": top["link"],
                    "explanation": f"Mixed or Misleading: Retrieved news reports present conflicting or disputed details regarding the claim.",
                    "evidence": top["title"]
                }
            elif conflicting_matches and not real_matches:
                top = conflicting_matches[0]
                return {
                    "status": "Contradicted",
                    "title": top["title"],
                    "source": top["source"],
                    "link": top["link"],
                    "explanation": f"Claim Contradicted by Evidence: Reporting from {top['source']} indicates contradictory dates or factual details.",
                    "evidence": top["title"]
                }

            # Check for Supported Reporting
            if real_matches:
                real_matches.sort(key=lambda x: x["overlap"], reverse=True)
                top = real_matches[0]
                return {
                    "status": "Supported",
                    "title": top["title"],
                    "source": top["source"],
                    "link": top["link"],
                    "explanation": f"The claim is Supported by credible reporting from {top['source']}.",
                    "evidence": top["title"]
                }
        except Exception:
            pass

    # Fallback: Unable to Verify without guessing
    return {
        "status": "Unable to Verify",
        "title": "No Verified Reference Record Found",
        "source": "Reference Knowledge & Credible News Feeds",
        "link": "",
        "explanation": "No verified reference sources or credible news reporting could confirm or refute this claim. The system avoids guessing factual truth.",
        "evidence": ""
    }


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json() or {}
        text = data.get("text", "")
    else:
        data = request.form
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

    # 1. Independent Factual Verification Layer
    # Evaluates facts against official government references, encyclopedic records, and credible media.
    verification_result = verify_factual_claim_pipeline(text)
    verification_status = verification_result.get("status", "Unable to Verify")

    # 2. Machine Learning Text Classifier
    # Evaluates stylistic features and vocabulary distributions.
    features = vectorizer.transform([cleaned_text])
    vocab_matches = int(features.nnz)
    word_count = len(text.split())

    raw_pred = int(model.predict(features)[0])
    ml_style_result = "Real News" if raw_pred == 1 else "Fake News"

    fake_prob = 0.0
    real_prob = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        fake_prob = round(float(probabilities[0]) * 100, 2)
        real_prob = round(float(probabilities[1]) * 100, 2)
        confidence = round(max(probabilities) * 100, 2)
    else:
        confidence = 100.0

    is_short_text = word_count < 20

    # Final Verdict Integration:
    # If authoritative reference evidence directly CONTRADICTS the claim,
    # the final overall verdict is Fake News (Disproven Claim).
    # If the claim is corroborated/supported by official reference/media, it is Real News.
    # Otherwise, it relies on the ML linguistic classification.
    if verification_status == "Contradicted":
        result = "Fake News"
        verdict_override = True
        verdict_reason = "Classified as Fake News because the claim is contradicted by verified reference evidence."
    elif verification_status == "Supported":
        result = "Real News"
        verdict_override = False
        verdict_reason = "Corroborated by verified reference evidence."
    else:
        result = ml_style_result
        verdict_override = False
        verdict_reason = "Based on machine learning linguistic pattern analysis (Reference evidence unable to verify)."

    verified_info = {
        "status": verification_status,
        "type": verification_status.upper().replace(" ", "_"),
        "title": verification_result.get("title", ""),
        "source": verification_result.get("source", ""),
        "link": verification_result.get("link", ""),
        "explanation": verification_result.get("explanation", ""),
        "description": verification_result.get("explanation", ""),
        "evidence": verification_result.get("evidence", "")
    }

    source_type = (data.get("source_type") if request.is_json else request.form.get("source_type")) or "Custom News Article"

    if request.is_json:
        return jsonify({
            "prediction": result,
            "ml_style_prediction": ml_style_result,
            "verdict_override": verdict_override,
            "verdict_reason": verdict_reason,
            "raw_class": raw_pred,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "word_count": word_count,
            "is_short_text": is_short_text,
            "vocab_matches": vocab_matches,
            "verification_status": verification_status,
            "verified_info": verified_info,
            "source_type": source_type
        })

    return render_template(
        "result.html",
        prediction=result,
        ml_style_prediction=ml_style_result,
        verdict_override=verdict_override,
        verdict_reason=verdict_reason,
        raw_class=raw_pred,
        confidence=confidence,
        fake_probability=fake_prob,
        real_probability=real_prob,
        news_text=text,
        word_count=word_count,
        is_short_text=is_short_text,
        vocab_matches=vocab_matches,
        verification_status=verification_status,
        verified_info=verified_info,
        source_type=source_type
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)