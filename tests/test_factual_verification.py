import urllib.request
import json

test_cases = [
    {
        "name": "1. Correct Factual Claim",
        "text": "15th August is celebrated as Independence Day in India",
        "expected_status": "Supported",
        "notes": "Matches official Government of India record"
    },
    {
        "name": "2. Incorrect Date Claim",
        "text": "17th may is celebrated as independence day",
        "expected_status": "Contradicted",
        "notes": "Contradicts official Government of India date (15 August)"
    },
    {
        "name": "3. Unsupported Claim (Local / Private Event)",
        "text": "A local farmer found a rare blue crystal in his backyard",
        "expected_status": "Unable to Verify",
        "notes": "No authoritative record or news coverage found; avoids guessing"
    },
    {
        "name": "4. Irrelevant Evidence / Non-Matching Details",
        "text": "Ambani family celebrated an anniversary on a random weekend",
        "expected_status": "Unable to Verify",
        "notes": "Disallowed social media posts or loose keyword matches are rejected"
    },
    {
        "name": "5. Conflicting or Disputed Claim",
        "text": "The moon landing in 1969 was staged on a movie set",
        "expected_status": "Contradicted",
        "notes": "Flagged as debunked hoax / contradicted by encyclopedic records"
    },
    {
        "name": "6. Unavailable Source / Unverifiable Abstract Statement",
        "text": "Secret unknown committee meeting happened in undisclosed forest",
        "expected_status": "Unable to Verify",
        "notes": "Returns 'Unable to Verify' safely without fabricating sources"
    }
]

print("=" * 70)
print("RUNNING AUTOMATED FACTUAL VERIFICATION TEST SUITE")
print("=" * 70)

all_passed = True

for case in test_cases:
    req = urllib.request.Request(
        "http://127.0.0.1:5000/predict",
        data=json.dumps({"text": case["text"]}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        data = json.loads(res.read().decode("utf-8"))
        status = data.get("verification_status")
        ml_pred = data.get("prediction")
        conf = data.get("confidence")
        verified_info = data.get("verified_info") or {}

        passed = (status == case["expected_status"])
        if not passed:
            all_passed = False

        print(f"[{'PASS' if passed else 'FAIL'}] {case['name']}")
        print(f"  Claim Input: \"{case['text']}\"")
        print(f"  ML Classification: {ml_pred} (Confidence: {conf}%)")
        print(f"  Verification Status: {status} (Expected: {case['expected_status']})")
        print(f"  Source: {verified_info.get('source')}")
        print(f"  Explanation: {verified_info.get('explanation')}")
        if verified_info.get("evidence"):
            print(f"  Evidence: {verified_info.get('evidence')[:130]}...")
        print("-" * 70)

print("\nFINAL TEST SUITE SUMMARY:", "ALL 6 TEST CASES PASSED!" if all_passed else "SOME CASES FAILED.")
print("=" * 70)
