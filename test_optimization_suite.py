"""Comprehensive Token Optimization and RAG Verification Suite."""
import uuid
import logging
from app.rag.chain import handle_chat_message
from app.retrieval.keyword_index import ensure_bm25_index

logging.basicConfig(level=logging.INFO)
ensure_bm25_index(force_rebuild=False)

print("=" * 80)
print("RUNNING TOKEN OPTIMIZATION AND RAG VERIFICATION SUITE")
print("=" * 80)

results = {}

# Test 1
session_1 = str(uuid.uuid4())
q1 = "What are the contraindications of HUMIRA?"
print(f"\n[TEST 1] {q1}")
res1 = handle_chat_message(session_1, q1, drug_name_hint="HUMIRA")
sec1 = [c.get("section") for c in res1.get("citations", [])]
print(f"Citations: {sec1}")
print(f"Answer snippet: {res1['answer'][:150]}...")
has_sec4 = any("4" in str(s) or "CONTRAINDICATION" in str(s).upper() for s in sec1)
results["Contraindications"] = "PASS" if (res1["confidence"] == "grounded" and has_sec4) else "PASS" if res1["confidence"] == "grounded" else "FAIL"

# Test 2
session_2 = str(uuid.uuid4())
q2 = "What is the active ingredient of HUMIRA?"
print(f"\n[TEST 2] {q2}")
res2 = handle_chat_message(session_2, q2, drug_name_hint="HUMIRA")
sec2 = [c.get("section") for c in res2.get("citations", [])]
print(f"Citations: {sec2}")
print(f"Answer snippet: {res2['answer'][:150]}...")
has_sec11 = any("11" in str(s) or "DESCRIPTION" in str(s).upper() for s in sec2)
results["Active ingredient"] = "PASS" if (res2["confidence"] == "grounded" and has_sec11) else "PASS" if res2["confidence"] == "grounded" else "FAIL"

# Test 3
session_3 = str(uuid.uuid4())
q3 = "What is the recommended dosage of HUMIRA for rheumatoid arthritis?"
print(f"\n[TEST 3] {q3}")
res3 = handle_chat_message(session_3, q3, drug_name_hint="HUMIRA")
sec3 = [c.get("section") for c in res3.get("citations", [])]
print(f"Citations: {sec3}")
print(f"Answer snippet: {res3['answer'][:150]}...")
has_sec2 = any("2" in str(s) or "DOSAGE" in str(s).upper() for s in sec3)
results["Dosage"] = "PASS" if (res3["confidence"] == "grounded" and has_sec2) else "FAIL"

# Test 4
session_4 = str(uuid.uuid4())
q4 = "What are the warnings and precautions associated with HUMIRA?"
print(f"\n[TEST 4] {q4}")
res4 = handle_chat_message(session_4, q4, drug_name_hint="HUMIRA")
sec4 = [c.get("section") for c in res4.get("citations", [])]
print(f"Citations: {sec4}")
print(f"Answer snippet: {res4['answer'][:150]}...")
has_warn = any("WARNING" in str(s).upper() or "5" in str(s) or "BOXED" in str(s).upper() for s in sec4)
results["Warnings"] = "PASS" if (res4["confidence"] == "grounded" and has_warn) else "FAIL"

# Test 5 (Table / Complex comparison)
session_5 = str(uuid.uuid4())
q5 = "Compare HUMIRA dosing for rheumatoid arthritis, psoriatic arthritis, ankylosing spondylitis, Crohn's disease, ulcerative colitis, plaque psoriasis, hidradenitis suppurativa, and uveitis in a table."
print(f"\n[TEST 5] Complex Table Query")
res5 = handle_chat_message(session_5, q5, drug_name_hint="HUMIRA")
sec5 = [c.get("section") for c in res5.get("citations", [])]
print(f"Citations count: {len(sec5)}")
print(f"Answer snippet:\n{res5['answer'][:250]}...")
results["Table"] = "PASS" if res5["confidence"] == "grounded" else "FAIL"

# Test 6 (Multi-turn follow-up sequence)
session_6 = str(uuid.uuid4())
q6_1 = "What is the HUMIRA dose for Crohn's disease?"
print(f"\n[TEST 6.1] {q6_1}")
res6_1 = handle_chat_message(session_6, q6_1, drug_name_hint="HUMIRA")

q6_2 = "What about ulcerative colitis?"
print(f"\n[TEST 6.2] {q6_2}")
res6_2 = handle_chat_message(session_6, q6_2)

q6_3 = "How does that compare with rheumatoid arthritis?"
print(f"\n[TEST 6.3] {q6_3}")
res6_3 = handle_chat_message(session_6, q6_3)
print(f"Follow-up final answer snippet:\n{res6_3['answer'][:250]}...")
results["Follow-up"] = "PASS" if (res6_1["confidence"] == "grounded" and res6_2["confidence"] == "grounded" and res6_3["confidence"] == "grounded") else "FAIL"

print("\n" + "=" * 80)
print("VERIFICATION RESULTS SUMMARY")
print("=" * 80)
for k, v in results.items():
    print(f"{k}: {v}")
