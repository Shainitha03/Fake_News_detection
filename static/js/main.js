let currentSamples = [];

function loadNewSamples() {
    const dropdown = document.getElementById("sampleDropdown");
    if (!dropdown) return;
    dropdown.innerHTML = '<option value="">⏳ Loading sample news articles...</option>';

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
    dropdown.innerHTML = '<option value="">-- Select a sample news article --</option>';

    currentSamples.forEach((sample, index) => {
        const option = document.createElement("option");
        option.value = index;
        const shortTitle = sample.title.length > 80 ? sample.title.substring(0, 77) + "..." : sample.title;
        option.textContent = `${index + 1}. ${shortTitle}`;
        dropdown.appendChild(option);
    });
}

function pasteSample() {
    const dropdown = document.getElementById("sampleDropdown");
    const index = dropdown.value;
    if (index !== "" && currentSamples[index]) {
        document.getElementById("newsText").value = currentSamples[index].text;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    loadNewSamples();
});

function validateForm() {
    const newsText = document.getElementById("newsText").value.trim();
    if (!newsText) {
        alert("⚠️ Please enter or load a news article.");
        return false;
    }
    
    // Optional: show some loading UI before page unloads
    document.getElementById("newsForm").style.opacity = "0.5";
    return true;
}

function clearText() {
    document.getElementById("newsText").value = "";
    const dropdown = document.getElementById("sampleDropdown");
    if (dropdown) dropdown.value = "";

    const result = document.getElementById("result");
    if (result) {
        result.style.display = "none";
        result.innerHTML = "";
    }
}