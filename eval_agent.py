"""
Automated evaluation for your agent.

Until now, testing meant manually typing questions into Flutter and reading
the answers yourself. This script does the same job automatically: it runs
a fixed list of test cases, checks whether each answer contains the facts
it should, and gives you a pass/fail report at the end.

This is a real skill called "evals" in AI development — production AI teams
run test suites like this constantly to catch regressions when they change
a prompt, swap a model, or update a tool.

Run this against server_rag.py (start it first, in a separate terminal):
    uvicorn server_rag:app --host 0.0.0.0 --port 8000

Then run this script:
    python eval_agent.py
"""

import requests
import time

BASE_URL = "http://127.0.0.1:8000/chat"

# Each test case: a question, and a list of key facts that MUST appear
# somewhere in the answer for it to count as a pass. Case-insensitive.
TEST_CASES = [
    {
        "name": "Static knowledge (no tool needed)",
        "question": "What is the capital of France?",
        "must_contain": ["paris"],
        "should_not_search": True,  # expect NO sources/tool use
    },
    {
        "name": "Math via calculator",
        "question": "What is 847 multiplied by 293?",
        "must_contain": ["248171", "248,171"],  # accept either format
        "match_mode": "any",
    },
    {
        "name": "RAG — tuition fees",
        "question": "How much does the BS Computer Science program cost per semester?",
        "must_contain": ["145,000", "145000"],  # allow either format
        "match_mode": "any",  # pass if ANY of these appear, not all
    },
    {
        "name": "RAG — Vice Chancellor name",
        "question": "Who is the Vice Chancellor of AFUT?",
        "must_contain": ["imran chaudhry"],
    },
    {
        "name": "RAG — AI program launch year",
        "question": "When was the AI degree program launched?",
        "must_contain": ["2023"],
    },
]


def run_test(client, session_id, test_case):
    """Send one test question and check the answer against expectations."""
    response = requests.post(
        BASE_URL,
        json={"message": test_case["question"], "session_id": session_id},
        timeout=120  # local models can be slow — give it time
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("answer", "")
    sources = data.get("sources", [])

    answer_lower = answer.lower()

    match_mode = test_case.get("match_mode", "all")
    required = [f.lower() for f in test_case["must_contain"]]

    if match_mode == "any":
        passed = any(fact in answer_lower for fact in required)
    else:
        passed = all(fact in answer_lower for fact in required)

    # Optional check: some tests expect NO tool/search to have happened
    if test_case.get("should_not_search") and sources:
        passed = False

    return {
        "name": test_case["name"],
        "question": test_case["question"],
        "answer": answer,
        "passed": passed,
        "sources_used": len(sources) > 0,
    }


def main():
    print("Running evaluation suite against your agent...\n")
    results = []

    for i, test_case in enumerate(TEST_CASES):
        session_id = f"eval_test_{i}"  # fresh session per test, no shared memory
        print(f"Running: {test_case['name']}...")
        result = run_test(requests, session_id, test_case)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  → {status}")
        time.sleep(1)  # small pause between requests

    # --- Report ---
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n{status} — {r['name']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:150]}{'...' if len(r['answer']) > 150 else ''}")

    print("\n" + "=" * 60)
    print(f"RESULT: {passed_count}/{total} tests passed ({passed_count/total*100:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()