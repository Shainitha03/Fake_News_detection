let currentSamples = [];

function loadNewSamples() {
    const dropdown = document.getElementById("sampleDropdown");
    if (!dropdown) return;
    dropdown.innerHTML = '<option value="">⏳ Loading sample news articles from dataset...</option>';

    fetch("/samples_list")
        .then(response => {
            if (!response.ok) {
                throw new Error("HTTP error " + response.status);
            }
            return response.json();
        })
        .then(data => {
            if (data.samples && data.samples.length > 0) {
                currentSamples = data.samples;
                renderSamplesDropdown();
            } else {
                dropdown.innerHTML = '<option value="">No samples available</option>';
            }
        })
        .catch(err => {
            console.error("Error fetching samples:", err);
            dropdown.innerHTML = '<option value="">⚠️ Failed to load samples (click Refresh)</option>';
        });
}

function renderSamplesDropdown() {
    const dropdown = document.getElementById("sampleDropdown");
    if (!dropdown) return;
    dropdown.innerHTML = '<option value="">-- Select a sample news article from dataset --</option>';

    currentSamples.forEach((sample, index) => {
        const option = document.createElement("option");
        option.value = index;
        const tag = sample.category ? `[${sample.category}] ` : "";
        const shortTitle = sample.title.length > 70 ? sample.title.substring(0, 67) + "..." : sample.title;
        option.textContent = `${index + 1}. ${tag}${shortTitle}`;
        dropdown.appendChild(option);
    });
}

function pasteSample() {
    const dropdown = document.getElementById("sampleDropdown");
    const index = dropdown.value;
    const datasetTextArea = document.getElementById("datasetNewsText");
    if (index !== "" && currentSamples[index] && datasetTextArea) {
        datasetTextArea.value = currentSamples[index].text;
    }
}

function switchMainTab(mode) {
    const tabCustomBtn = document.getElementById("tabCustomBtn");
    const tabDatasetBtn = document.getElementById("tabDatasetBtn");

    const customSection = document.getElementById("customSection");
    const datasetSection = document.getElementById("datasetSection");

    if (mode === "dataset") {
        if (tabDatasetBtn) tabDatasetBtn.classList.add("active");
        if (tabCustomBtn) tabCustomBtn.classList.remove("active");
        if (datasetSection) datasetSection.style.display = "block";
        if (customSection) customSection.style.display = "none";
        const dropdown = document.getElementById("sampleDropdown");
        if (dropdown && dropdown.options.length <= 1) {
            loadNewSamples();
        }
    } else {
        if (tabCustomBtn) tabCustomBtn.classList.add("active");
        if (tabDatasetBtn) tabDatasetBtn.classList.remove("active");
        if (customSection) customSection.style.display = "block";
        if (datasetSection) datasetSection.style.display = "none";
        const customText = document.getElementById("customNewsText");
        if (customText) customText.focus();
    }
}

function validateCustomNewsForm() {
    const customText = document.getElementById("customNewsText");
    const text = customText ? customText.value.trim() : "";

    if (!text) {
        alert("⚠️ Please paste or type your news article before clicking 'Analyze My News'.");
        if (customText) customText.focus();
        return false;
    }

    const form = document.getElementById("customNewsForm");
    if (form) form.style.opacity = "0.6";
    return true;
}

function validateDatasetForm() {
    const datasetText = document.getElementById("datasetNewsText");
    const text = datasetText ? datasetText.value.trim() : "";

    if (!text) {
        alert("⚠️ Please select an article from the dataset dropdown above or paste article text.");
        if (datasetText) datasetText.focus();
        return false;
    }

    const form = document.getElementById("datasetNewsForm");
    if (form) form.style.opacity = "0.6";
    return true;
}

function clearCustomText() {
    const customText = document.getElementById("customNewsText");
    if (customText) {
        customText.value = "";
        customText.focus();
    }
}

function clearDatasetText() {
    const datasetText = document.getElementById("datasetNewsText");
    if (datasetText) datasetText.value = "";
    const dropdown = document.getElementById("sampleDropdown");
    if (dropdown) dropdown.value = "";
}

document.addEventListener("DOMContentLoaded", () => {
    loadNewSamples();
});