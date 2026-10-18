"""
Comprehensive Automated Test Suite for Fake News Detection with Factual Verification.

Covers:
1. Correct factual dates (e.g. 15 August Independence Day in India, 14 July Bastille Day in France)
2. Incorrect factual dates (e.g. 17 May Independence Day in India, 20 August Bastille Day in France)
3. Real news articles/claims
4. Fake news articles/claims
5. Unsupported claims (e.g. local farmer found rare blue crystal)
6. Empty input and whitespace-only input
7. Unavailable verification evidence
8. Disagreement handling between ML style model and factual verification
"""

import sys
import json
import unittest

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app import app, verify_factual_claim_pipeline, clean_text


class TestFactualVerificationAndPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    # -------------------------------------------------------------------------
    # 1. Correct Factual Dates
    # -------------------------------------------------------------------------
    def test_correct_factual_date_india_independence(self):
        text = "15 August is celebrated as Independence Day in India."
        result = verify_factual_claim_pipeline(text)
        self.assertEqual(result["status"], "Supported")
        self.assertIn("15 August", result["explanation"])

        res = self.client.post("/predict", json={"text": text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["prediction"], "Real News")
        self.assertEqual(data["factual_verification"], "SUPPORTED")
        self.assertIn("REAL NEWS", data["prediction_display"])

    def test_correct_factual_date_france_bastille(self):
        text = "Bastille Day is celebrated on 14 July in France."
        result = verify_factual_claim_pipeline(text)
        self.assertEqual(result["status"], "Supported")
        self.assertIn("14 July", result["explanation"])

    # -------------------------------------------------------------------------
    # 2. Incorrect Factual Dates (Example input & General cases)
    # -------------------------------------------------------------------------
    def test_incorrect_factual_date_india_independence_example(self):
        # Example requirement: 17 May is celebrated as Independence Day in India
        text = "17 May is celebrated as Independence Day in India."
        result = verify_factual_claim_pipeline(text)
        self.assertEqual(result["status"], "Contradicted")
        self.assertIn("15 August", result["explanation"])
        self.assertIn("17 May", result["explanation"])

        res = self.client.post("/predict", json={"text": text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["prediction"], "Fake News")
        self.assertEqual(data["prediction_display"], "🔴 Prediction: FAKE NEWS ❌")
        self.assertEqual(
            data["reason"],
            "In India, Independence Day is celebrated on 15 August, not 17 May."
        )
        self.assertEqual(data["factual_verification"], "CONTRADICTED")
        # ML style may predict Real News for formal writing, so disagreement explanation is present
        if data["ml_style_prediction"] == "Real News":
            self.assertIsNotNone(data["disagreement_explanation"])
            self.assertIn("Model vs Evidence Disagreement", data["disagreement_explanation"])

    def test_incorrect_factual_date_republic_day(self):
        text = "Republic Day in India is celebrated on 10 November."
        result = verify_factual_claim_pipeline(text)
        self.assertEqual(result["status"], "Contradicted")
        self.assertIn("26 January", result["explanation"])
        self.assertIn("10 November", result["explanation"])

    # -------------------------------------------------------------------------
    # 3. Real News Examples
    # -------------------------------------------------------------------------
    def test_real_news_reporting(self):
        text = "The Supreme Court delivered a constitutional bench ruling outlining legal procedures."
        res = self.client.post("/predict", json={"text": text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data["prediction"], ["Real News", "Fake News"])
        self.assertIn("prediction_display", data)
        self.assertIn("reason", data)

    # -------------------------------------------------------------------------
    # 4. Fake News Examples / Debunked Claims
    # -------------------------------------------------------------------------
    def test_fake_news_debunked_claim(self):
        text = "The moon landing in 1969 was staged on a movie set."
        result = verify_factual_claim_pipeline(text)
        self.assertEqual(result["status"], "Contradicted")

        res = self.client.post("/predict", json={"text": text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["prediction"], "Fake News")
        self.assertEqual(data["factual_verification"], "CONTRADICTED")

    # -------------------------------------------------------------------------
    # 5. Unsupported Claims (Private/Unverified Events)
    # -------------------------------------------------------------------------
    def test_unsupported_claim(self):
        text = "A local farmer found a rare blue crystal in his backyard."
        result = verify_factual_claim_pipeline(text)
        self.assertEqual(result["status"], "Unable to Verify")
        self.assertEqual(
            result["explanation"],
            "Unable to verify this claim with the available evidence."
        )

        res = self.client.post("/predict", json={"text": text})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["factual_verification"], "UNABLE TO VERIFY")
        self.assertIn("Unable to verify this claim with the available evidence", data["reason"])
        # Does not override ML model when evidence is unavailable
        self.assertFalse(data["verdict_override"])

    # -------------------------------------------------------------------------
    # 6. Empty Input & Whitespace-Only Input
    # -------------------------------------------------------------------------
    def test_empty_input(self):
        res = self.client.post("/predict", json={"text": ""})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)
        self.assertIn("Please enter a news article", data["error"])

    def test_whitespace_only_input(self):
        res = self.client.post("/predict", json={"text": "     \n\t  "})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    # -------------------------------------------------------------------------
    # 7. Unavailable Verification Evidence
    # -------------------------------------------------------------------------
    def test_unavailable_verification_evidence(self):
        text = "Secret unknown committee meeting happened in undisclosed forest with unknown participants."
        result = verify_factual_claim_pipeline(text)
        self.assertEqual(result["status"], "Unable to Verify")
        self.assertEqual(
            result["explanation"],
            "Unable to verify this claim with the available evidence."
        )

    # -------------------------------------------------------------------------
    # 8. Preservation of Preprocessing and Clean Text
    # -------------------------------------------------------------------------
    def test_clean_text_preservation(self):
        text = "Breaking News: 15 August is celebrated annually!"
        cleaned = clean_text(text)
        self.assertIsInstance(cleaned, str)
        self.assertIn("celebr", cleaned)

    # -------------------------------------------------------------------------
    # 9. Model Label Mapping Verification
    # -------------------------------------------------------------------------
    def test_model_label_mapping(self):
        import joblib
        model = joblib.load("models/model.pkl")
        # In this dataset/training pipeline: 0 = Fake, 1 = Real
        self.assertEqual(list(model.classes_), [0, 1])

    # -------------------------------------------------------------------------
    # 10. Frontend and Backend Result Consistency
    # -------------------------------------------------------------------------
    def test_frontend_backend_result_consistency(self):
        text = "17 May is celebrated as Independence Day in India."
        # HTML request (frontend)
        html_res = self.client.post("/predict", data={"text": text})
        self.assertEqual(html_res.status_code, 200)
        html_content = html_res.get_data(as_text=True)

        # JSON request (backend API)
        json_res = self.client.post("/predict", json={"text": text})
        self.assertEqual(json_res.status_code, 200)
        json_data = json_res.get_json()

        # Check consistency between JSON payload and HTML rendering
        self.assertEqual(json_data["prediction"], "Fake News")
        self.assertEqual(json_data["factual_verification"], "CONTRADICTED")
        self.assertIn("FAKE NEWS", html_content)
        self.assertIn("CONTRADICTED", html_content)
        self.assertIn("In India, Independence Day is celebrated on 15 August, not 17 May.", html_content)


if __name__ == "__main__":
    unittest.main()
