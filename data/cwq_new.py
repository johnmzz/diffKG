import json
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "cwq.json"
ANSWER_MATCH_OUTPUT = "cwq_answer_match.json"
ANSWER_NO_MATCH_OUTPUT = "cwq_answer_nomatch.json"
UPDATED_QA_FILE = "cwq_new.json"

FUZZY_THRESHOLD = 0.75
FUZZY_TOP_K = 10


def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_english_label(mid):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?name
    FROM <http://freebase.com>
    WHERE {{
        ns:{mid} ns:type.object.name ?name .
        FILTER (lang(?name) = "en")
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            return bindings[0]["name"]["value"]
    except Exception as e:
        return None
    return None

def run_sparql_query(sparql_str):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(sparql_str)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        answer_mids = []
        for row in bindings:
            if "x" in row:
                uri = row["x"]["value"]
                if uri.startswith("http://rdf.freebase.com/ns/"):
                    mid = uri.replace("http://rdf.freebase.com/ns/", "")
                    answer_mids.append(mid)
        return answer_mids
    except Exception:
        return []

def main():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print("🔍 Running SPARQL for answers...")
    updated_data = []
    matched = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {}
        for qa in qa_data:
            sparql_query = qa.get("sparql", "")
            futures[executor.submit(run_sparql_query, sparql_query)] = qa

        for future in as_completed(futures):
            qa = futures[future]
            answer_text = qa.get("answer", "").strip()
            mids = future.result()
            answer_mid, answer_label = None, None
            for mid in mids:
                label = get_english_label(mid)
                if label and label.lower() == answer_text.lower():
                    answer_mid = mid
                    answer_label = label
                    matched += 1
                    break
            if answer_mid and answer_label:
                qa["answer"] = {answer_mid: answer_label}
            else:
                qa["answer"] = {}
                failed += 1
            updated_data.append(qa)

    with open(UPDATED_QA_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ CWQ updated with answer MIDs → saved to {UPDATED_QA_FILE}")
    print(f"✅ Matched answers: {matched}")
    print(f"❌ Failed to match answers: {failed}")

if __name__ == "__main__":
    main()
