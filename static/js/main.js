/**
 * VERITAS AI - Modern Fake News Detection & Fact Verification
 * Interactive Client-Side Engine
 * Features: XAI Token Attribution, URL Scraper, Voice Dictation, Dark/Light Mode
 */

// State Management
let allDatasetSamples = [];
let activeDatasetFilter = 'all';
let currentLiveSource = 'bbc';
let cachedLiveArticles = {};
let speechRecognizer = null;
let isRecordingVoice = false;

// Preset Sample Claims for Quick Testing
const PRESET_CLAIMS = [
    {
        title: "🌑 Apollo Moon Landing Hoax",
        text: "NASA whistleblower admits on deathbed that the 1969 Apollo moon landings were staged on an abandoned Hollywood movie set using special lighting effects and wires to fool the entire world."
    },
    {
        title: "🏛️ Official Indian Republic Day",
        text: "India celebrates Republic Day officially on 26 January to mark the adoption of the Constitution of India in 1950, featuring grand parades at Kartavya Path in New Delhi."
    },
    {
        title: "🚀 NASA James Webb Discovery",
        text: "Astronomers utilizing the James Webb Space Telescope have discovered water vapor and carbon dioxide signatures in the atmosphere of a habitable-zone exoplanet located 120 light years away."
    },
    {
        title: "💊 Miracle Herb Cancer Cure",
        text: "Secret ancient dandelion root extract cures 98 percent of advanced cancer tumors in 48 hours, but big pharmaceutical companies and doctors are hiding it from hospitals to protect billion-dollar drug profits."
    },
    {
        title: "🇮🇳 Indian Independence Day",
        text: "Government of India gazette records confirm Independence Day is celebrated annually on 15 August to commemorate India's freedom from British colonial rule on 15 August 1947."
    }
];

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initPresets();
    initTextareaAnalytics();
    initSpeechRecognition();
});

// ==========================================================================
// 1. Theme Management (Dark / Light Mode)
// ==========================================================================
function initTheme() {
    const savedTheme = localStorage.getItem("veritas_theme") || "dark";
    applyTheme(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(newTheme);
    localStorage.setItem("veritas_theme", newTheme);
    showToast(`Switched to ${newTheme === "light" ? "Clean Editorial Light" : "Cyber Dark"} mode`);
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const themeBtn = document.getElementById("themeToggleBtn");
    if (themeBtn) {
        if (theme === "light") {
            themeBtn.innerHTML = "🌙 <span>Dark Mode</span>";
        } else {
            themeBtn.innerHTML = "☀️ <span>Light Mode</span>";
        }
    }
}

// ==========================================================================
// 2. Presets & Text Analytics
// ==========================================================================
function initPresets() {
    const container = document.getElementById("presetsScroll");
    if (!container) return;

    container.innerHTML = "";
    PRESET_CLAIMS.forEach(preset => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "preset-chip";
        chip.innerHTML = preset.title;
        chip.onclick = () => {
            const textarea = document.getElementById("customNewsText");
            if (textarea) {
                textarea.value = preset.text;
                updateTextAnalytics();
                showToast(`Loaded: ${preset.title}`);
                textarea.focus();
            }
        };
        container.appendChild(chip);
    });
}

function initTextareaAnalytics() {
    const textarea = document.getElementById("customNewsText");
    if (!textarea) return;

    textarea.addEventListener("input", updateTextAnalytics);
    updateTextAnalytics();
}

function updateTextAnalytics() {
    const textarea = document.getElementById("customNewsText");
    if (!textarea) return;

    const text = textarea.value.trim();
    const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
    const chars = text.length;
    const readingTime = Math.max(1, Math.ceil(words / 200));

    const wordEl = document.getElementById("analyticsWordCount");
    const charEl = document.getElementById("analyticsCharCount");
    const readEl = document.getElementById("analyticsReadTime");

    if (wordEl) wordEl.textContent = words;
    if (charEl) charEl.textContent = chars;
    if (readEl) readEl.textContent = `${readingTime} min`;

    const hint = document.getElementById("textGuidanceHint");
    if (hint) {
        if (words === 0) {
            hint.innerHTML = "💡 <em>Type, paste, dictate, or scrape news text above.</em>";
        } else if (words < 15) {
            hint.innerHTML = "⚠️ <span style='color: #f59e0b;'>Short text: Model accuracy is highest with 20+ words.</span>";
        } else {
            hint.innerHTML = "✅ <span style='color: #10b981;'>Optimal sample length for NLP vectorization.</span>";
        }
    }
}

async function pasteFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        const textarea = document.getElementById("customNewsText");
        if (textarea && text) {
            textarea.value = text;
            updateTextAnalytics();
            showToast("Pasted from clipboard!");
        }
    } catch (err) {
        showToast("Please allow clipboard access or use Ctrl+V");
    }
}

function clearCustomText() {
    const textarea = document.getElementById("customNewsText");
    if (textarea) {
        textarea.value = "";
        updateTextAnalytics();
        textarea.focus();
    }
    const resultBox = document.getElementById("liveResultContainer");
    if (resultBox) resultBox.style.display = "none";
}

// ==========================================================================
// 3. News URL / Link Scraper Feature
// ==========================================================================
function toggleUrlInput(show) {
    const urlBox = document.getElementById("urlScraperBox");
    if (!urlBox) return;
    if (show) {
        urlBox.style.display = "block";
        const urlInput = document.getElementById("newsUrlInput");
        if (urlInput) urlInput.focus();
    } else {
        urlBox.style.display = "none";
    }
}

async function scrapeAndAnalyzeUrl() {
    const urlInput = document.getElementById("newsUrlInput");
    const url = urlInput ? urlInput.value.trim() : "";

    if (!url) {
        showToast("⚠️ Please enter a news article link.");
        if (urlInput) urlInput.focus();
        return;
    }

    const scrapeBtn = document.getElementById("scrapeUrlBtn");
    if (scrapeBtn) {
        scrapeBtn.disabled = true;
        scrapeBtn.innerHTML = "⏳ Scraping...";
    }

    showToast("📡 Fetching and parsing article from URL...");

    try {
        const response = await fetch("/scrape_url", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Failed to scrape URL");
        }

        const textarea = document.getElementById("customNewsText");
        if (textarea && data.text) {
            textarea.value = data.text;
            updateTextAnalytics();
            showToast(`Scraped "${data.title.substring(0, 40)}..." from ${data.source}!`);
            
            toggleUrlInput(false);
            runInstantAnalysis();
        }
    } catch (err) {
        console.error("Scraper error:", err);
        showToast("⚠️ " + err.message);
    } finally {
        if (scrapeBtn) {
            scrapeBtn.disabled = false;
            scrapeBtn.innerHTML = "⚡ Scrape &amp; Analyze";
        }
    }
}

// ==========================================================================
// 4. Voice / Speech-to-Text Input Feature
// ==========================================================================
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const voiceBtn = document.getElementById("voiceDictateBtn");

    if (!SpeechRecognition) {
        if (voiceBtn) {
            voiceBtn.title = "Speech recognition is supported in Chrome, Edge, and Safari";
        }
        return;
    }

    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = "en-US";

    speechRecognizer.onstart = () => {
        isRecordingVoice = true;
        updateVoiceButtonUI();
        showToast("🎙️ Microphone active: speak now...");
    };

    speechRecognizer.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            transcript += event.results[i][0].transcript;
        }

        const textarea = document.getElementById("customNewsText");
        if (textarea && transcript.trim()) {
            textarea.value = (textarea.value.trim() ? textarea.value.trim() + " " : "") + transcript;
            updateTextAnalytics();
        }
    };

    speechRecognizer.onerror = (event) => {
        console.warn("Speech recognition error:", event.error);
        isRecordingVoice = false;
        updateVoiceButtonUI();
        if (event.error === "not-allowed") {
            showToast("⚠️ Microphone access denied. Check browser permissions.");
        }
    };

    speechRecognizer.onend = () => {
        isRecordingVoice = false;
        updateVoiceButtonUI();
    };
}

function toggleVoiceDictation() {
    if (!speechRecognizer) {
        showToast("⚠️ Speech Recognition is not supported in this browser. Please use Chrome or Edge.");
        return;
    }

    if (isRecordingVoice) {
        speechRecognizer.stop();
        isRecordingVoice = false;
        updateVoiceButtonUI();
        showToast("Voice input stopped.");
    } else {
        try {
            speechRecognizer.start();
        } catch (err) {
            console.error("Could not start speech recognition:", err);
        }
    }
}

function updateVoiceButtonUI() {
    const voiceBtn = document.getElementById("voiceDictateBtn");
    if (!voiceBtn) return;

    if (isRecordingVoice) {
        voiceBtn.classList.add("recording");
        voiceBtn.innerHTML = "🔴 Stop";
    } else {
        voiceBtn.classList.remove("recording");
        voiceBtn.innerHTML = "🎙️ Dictate";
    }
}

// ==========================================================================
// 5. Tab Navigation
// ==========================================================================
function switchMainTab(mode) {
    const tabButtons = document.querySelectorAll(".nav-tab-btn");
    tabButtons.forEach(btn => btn.classList.remove("active"));

    const panels = document.querySelectorAll(".mode-panel");
    panels.forEach(p => p.classList.remove("active-panel"));

    const targetBtn = document.getElementById(`tab-${mode}-btn`);
    const targetPanel = document.getElementById(`section-${mode}`);

    if (targetBtn) targetBtn.classList.add("active");
    if (targetPanel) targetPanel.classList.add("active-panel");

    if (mode === "dataset" && allDatasetSamples.length === 0) {
        loadDatasetSamples();
    } else if (mode === "radar" && Object.keys(cachedLiveArticles).length === 0) {
        loadLiveNews(currentLiveSource);
    } else if (mode === "metrics") {
        loadModelMetrics();
    }
}

// ==========================================================================
// 6. Instant In-Page Prediction & Scanning HUD
// ==========================================================================
async function runInstantAnalysis(event) {
    if (event) event.preventDefault();

    const textarea = document.getElementById("customNewsText");
    const text = textarea ? textarea.value.trim() : "";

    if (!text) {
        showToast("⚠️ Please enter news text or choose a preset!");
        if (textarea) textarea.focus();
        return;
    }

    const scanningHud = document.getElementById("scanningHud");
    const resultContainer = document.getElementById("liveResultContainer");
    const scanStep = document.getElementById("scanStepText");

    if (resultContainer) resultContainer.style.display = "none";
    if (scanningHud) scanningHud.style.display = "block";

    if (scanStep) scanStep.textContent = "Preprocessing: Lowercasing, removing watermarks & stemming...";

    const stepTimeout1 = setTimeout(() => {
        if (scanStep) scanStep.textContent = "Feature Vectorization: TF-IDF 50,000 N-Gram matrix match...";
    }, 450);

    const stepTimeout2 = setTimeout(() => {
        if (scanStep) scanStep.textContent = "Explainable AI: Calculating token attribution weights...";
    }, 900);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({
                text: text,
                source_type: "Custom Analysis"
            })
        });

        clearTimeout(stepTimeout1);
        clearTimeout(stepTimeout2);

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();

        setTimeout(() => {
            if (scanningHud) scanningHud.style.display = "none";
            renderLiveResult(data, text);
        }, 550);

    } catch (err) {
        clearTimeout(stepTimeout1);
        clearTimeout(stepTimeout2);
        if (scanningHud) scanningHud.style.display = "none";
        console.error("Prediction error:", err);
        showToast("Error processing prediction: " + err.message);
    }
}

// ==========================================================================
// 7. Render Live Results with Explainable AI (XAI)
// ==========================================================================
function renderLiveResult(data, originalText) {
    const container = document.getElementById("liveResultContainer");
    if (!container) return;

    const isReal = data.prediction === "Real News" || data.prediction === "REAL";
    const bannerClass = isReal ? "banner-real" : "banner-fake";
    const verdictIcon = isReal ? "🟢" : "🔴";
    const verdictTitle = isReal ? "REAL / CREDIBLE NEWS" : "POTENTIAL FAKE NEWS";
    const verdictSubtitle = isReal ? "Verified / Credible Content" : "Identified as Misinformation / Fake";

    const reasonText = data.reason || (isReal 
        ? "Content matches standard credible journalistic reporting practices. No factual contradictions were detected." 
        : "Flagged as potential fake news based on sensationalist linguistic markers and unverified assertions.");

    // Evidence quote or official reference
    let evidenceHtml = "";
    if (data.verified_info && (data.verified_info.evidence || data.verified_info.source)) {
        const info = data.verified_info;
        const isFactRef = (info.status === 'Supported' || info.status === 'Contradicted');
        evidenceHtml = `
            <div class="evidence-quote-box">
                <div class="evidence-quote-label">${isFactRef ? 'Authoritative Fact Reference:' : 'Corroborating Evidence &amp; Attributions:'}</div>
                ${info.evidence ? `<blockquote class="evidence-quote">"${escapeHtml(info.evidence)}"</blockquote>` : ""}
                ${info.source ? `<div class="evidence-source-meta"><strong>Source / Attribution:</strong> ${escapeHtml(info.source)}</div>` : ""}
                ${info.link ? `<div style="margin-top: 8px;"><a href="${info.link}" target="_blank" rel="noopener noreferrer" class="evidence-source-link">View Reference Document ↗</a></div>` : ""}
            </div>
        `;
    }

    container.innerHTML = `
        <div class="verdict-banner-modern ${bannerClass}">
            <div class="verdict-left">
                <div class="verdict-icon-bubble">${verdictIcon}</div>
                <div>
                    <div class="verdict-title-text">${verdictTitle}</div>
                    <div class="verdict-subtitle">${verdictSubtitle}</div>
                </div>
            </div>
        </div>

        <div class="reason-panel-card ${isReal ? 'reason-panel-real' : 'reason-panel-fake'}">
            <div class="reason-panel-header">
                <span class="reason-header-icon">${isReal ? '✅' : '⚠️'}</span>
                <h3 class="reason-header-title">${isReal ? 'Verification Summary' : 'Reason for Identification'}</h3>
            </div>
            <div class="reason-panel-body">
                <p class="reason-primary-text">${escapeHtml(reasonText)}</p>
                ${evidenceHtml}
            </div>
        </div>
    `;

    container.style.display = "block";
    container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function generateHighlightedTextHtml(text, highlightMap) {
    if (!text) return "";
    const words = text.split(/(\s+)/);

    return words.map(chunk => {
        if (/^\s+$/.test(chunk)) return chunk;
        const cleanWord = chunk.toLowerCase().replace(/[^a-z0-9]/g, "");

        // Check if root or word exists in highlight map
        let match = null;
        if (highlightMap[cleanWord]) {
            match = highlightMap[cleanWord];
        } else {
            // Check stem prefix
            for (const key in highlightMap) {
                if (cleanWord.startsWith(key) || key.startsWith(cleanWord)) {
                    match = highlightMap[key];
                    break;
                }
            }
        }

        if (match) {
            const isReal = match.impact === "REAL";
            const cssClass = isReal ? "xai-word-real" : "xai-word-fake";
            const sign = match.weight > 0 ? "+" : "";
            return `<span class="${cssClass}" title="Token '${cleanWord}': ${sign}${match.weight} (${match.impact})">${escapeHtml(chunk)}</span>`;
        }

        return escapeHtml(chunk);
    }).join("");
}

function copyReport(verdict, confidence, real, fake) {
    const text = `VERITAS AI News Analysis Report\nVerdict: ${verdict}\nConfidence: ${confidence}%\nLinguistic Distribution: Real ${real}% | Fake ${fake}%\nVerified with TF-IDF & Logistic Regression NLP`;
    navigator.clipboard.writeText(text).then(() => {
        showToast("Report copied to clipboard!");
    });
}

// ==========================================================================
// 8. Dataset Explorer Tab
// ==========================================================================
async function loadDatasetSamples() {
    const grid = document.getElementById("datasetGrid");
    if (!grid) return;

    grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">⏳ Loading benchmark dataset samples...</div>';

    try {
        const response = await fetch("/samples_list");
        if (!response.ok) throw new Error("HTTP " + response.status);
        const data = await response.json();
        allDatasetSamples = data.samples || [];
        renderDatasetCards();
    } catch (err) {
        console.error("Failed to load dataset:", err);
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: #f43f5e;">⚠️ Failed to load dataset samples. Click "Refresh Batch" to try again.</div>';
    }
}

function setDatasetFilter(filterType) {
    activeDatasetFilter = filterType;
    document.querySelectorAll(".dataset-filter-pills .filter-pill").forEach(p => p.classList.remove("active"));
    const activePill = document.getElementById(`filter-pill-${filterType}`);
    if (activePill) activePill.classList.add("active");
    renderDatasetCards();
}

function filterDatasetSearch() {
    renderDatasetCards();
}

function renderDatasetCards() {
    const grid = document.getElementById("datasetGrid");
    if (!grid) return;

    const searchInput = document.getElementById("datasetSearchInput");
    const query = searchInput ? searchInput.value.toLowerCase().trim() : "";

    let filtered = allDatasetSamples.filter(item => {
        if (activeDatasetFilter === 'real' && item.category !== 'REAL') return false;
        if (activeDatasetFilter === 'fake' && item.category !== 'FAKE') return false;
        if (query) {
            return (item.title && item.title.toLowerCase().includes(query)) ||
                   (item.text && item.text.toLowerCase().includes(query));
        }
        return true;
    });

    if (filtered.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">No matching articles found in current batch.</div>';
        return;
    }

    grid.innerHTML = "";
    filtered.forEach((sample, idx) => {
        const isReal = sample.category === 'REAL';
        const tagClass = isReal ? 'tag-real-badge' : 'tag-fake-badge';
        const tagLabel = isReal ? '🟢 Ground Truth: Real' : '🔴 Ground Truth: Fake';

        const card = document.createElement("div");
        card.className = "dataset-item-card";
        card.innerHTML = `
            <div>
                <div class="dataset-card-header">
                    <span class="dataset-category-tag ${tagClass}">${tagLabel}</span>
                    <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">#${idx + 1}</span>
                </div>
                <h4 class="dataset-card-title">${escapeHtml(sample.title || "Untitled Article")}</h4>
                <p class="dataset-card-preview">${escapeHtml(sample.text || "")}</p>
            </div>
            <div class="dataset-card-actions">
                <button type="button" class="btn btn-secondary" style="padding: 6px 14px; font-size: 12.5px;" onclick="loadSampleToDetector(${idx})">
                    ⚡ Test in Detector
                </button>
            </div>
        `;
        grid.appendChild(card);
    });
}

function loadSampleToDetector(index) {
    const sample = allDatasetSamples[index];
    if (!sample) return;

    switchMainTab('custom');
    const textarea = document.getElementById("customNewsText");
    if (textarea) {
        textarea.value = sample.text;
        updateTextAnalytics();
        textarea.focus();
        showToast("Article loaded! Click 'Instant ML Scan' to evaluate.");
    }
}

// ==========================================================================
// 9. Live RSS World News Radar Tab
// ==========================================================================
async function loadLiveNews(sourceKey) {
    currentLiveSource = sourceKey;

    document.querySelectorAll(".radar-source-selector .source-btn").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`source-btn-${sourceKey}`);
    if (activeBtn) activeBtn.classList.add("active");

    const grid = document.getElementById("rssNewsGrid");
    if (!grid) return;

    if (cachedLiveArticles[sourceKey]) {
        renderLiveArticles(cachedLiveArticles[sourceKey], sourceKey);
        return;
    }

    grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">📡 Fetching live global news feed via RSS...</div>';

    try {
        const response = await fetch(`/live_news?source=${sourceKey}`);
        if (!response.ok) throw new Error("HTTP " + response.status);
        const data = await response.json();
        const articles = data.articles || [];
        cachedLiveArticles[sourceKey] = articles;
        renderLiveArticles(articles, data.source || sourceKey);
    } catch (err) {
        console.error("Live news error:", err);
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: #f43f5e;">⚠️ Unable to fetch live news feed. Please try another source or check network.</div>';
    }
}

function renderLiveArticles(articles, sourceName) {
    const grid = document.getElementById("rssNewsGrid");
    if (!grid) return;

    if (!articles || articles.length === 0) {
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">No live headlines found.</div>';
        return;
    }

    grid.innerHTML = "";
    articles.forEach((item, idx) => {
        const card = document.createElement("div");
        card.className = "rss-card";
        card.innerHTML = `
            <div>
                <div class="rss-card-source">📡 ${escapeHtml(sourceName)} • Live Feed</div>
                <h4 class="rss-card-title">${escapeHtml(item.title || "Headline")}</h4>
                <p class="rss-card-desc">${escapeHtml(item.text || "")}</p>
            </div>
            <div style="display: flex; justify-content: flex-end; margin-top: 10px;">
                <button type="button" class="btn btn-secondary" style="padding: 6px 14px; font-size: 12.5px;" onclick="loadRssToDetector('${currentLiveSource}', ${idx})">
                    ⚡ Fact Check & Analyze
                </button>
            </div>
        `;
        grid.appendChild(card);
    });
}

function loadRssToDetector(sourceKey, index) {
    const list = cachedLiveArticles[sourceKey];
    if (!list || !list[index]) return;

    const item = list[index];
    switchMainTab('custom');
    const textarea = document.getElementById("customNewsText");
    if (textarea) {
        textarea.value = item.text;
        updateTextAnalytics();
        textarea.focus();
        showToast("Live news article loaded! Analyzing...");
        runInstantAnalysis();
    }
}

// ==========================================================================
// 10. Model Architecture & Performance Metrics Tab
// ==========================================================================
async function loadModelMetrics() {
    try {
        const res = await fetch("/model_metrics");
        if (!res.ok) return;
        const metrics = await res.json();

        if (metrics.accuracy) document.getElementById("gaugeAccuracy").textContent = `${metrics.accuracy}%`;
        if (metrics.precision) document.getElementById("gaugePrecision").textContent = `${metrics.precision}%`;
        if (metrics.recall) document.getElementById("gaugeRecall").textContent = `${metrics.recall}%`;
        if (metrics.f1_score) document.getElementById("gaugeF1").textContent = `${metrics.f1_score}%`;

        if (metrics.confusion_matrix && metrics.confusion_matrix.length === 2) {
            const [[tn, fp], [fn, tp]] = metrics.confusion_matrix;
            document.getElementById("matrixTN").textContent = Number(tn).toLocaleString();
            document.getElementById("matrixFP").textContent = Number(fp).toLocaleString();
            document.getElementById("matrixFN").textContent = Number(fn).toLocaleString();
            document.getElementById("matrixTP").textContent = Number(tp).toLocaleString();
        }
    } catch (err) {
        console.warn("Could not load model metrics:", err);
    }
}

// ==========================================================================
// 11. Toast Utility
// ==========================================================================
function showToast(message) {
    let toast = document.getElementById("toastNotice");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "toastNotice";
        toast.className = "toast-notice";
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add("show-toast");

    setTimeout(() => {
        toast.classList.remove("show-toast");
    }, 3200);
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/[&<>"']/g, function(m) {
        return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        }[m];
    });
}