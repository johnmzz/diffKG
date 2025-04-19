import json
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "cwq.json"
OUTPUT_FILE = "mismatch_report.json"
CWQ_NEW_FILE = "cwq_new.json"


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
    except Exception:
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

    total_qas = len(qa_data)
    total_entities = 0
    matched = 0
    failed = []
    updated_qa_data = []

    topic_1 = topic_more = topic_more_than_2 = answer_1 = answer_more = both_more = 0
    topic_more_than_2_ids = []

    answer_total = 0
    answer_matched = 0
    answer_mismatch = []

    print("🔍 Checking topic entity labels...")
    topic_check_tasks = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for qa in qa_data:
            topic_entities = qa.get("topic_entity", {})
            if len(topic_entities) == 1:
                topic_1 += 1
            elif len(topic_entities) > 1:
                topic_more += 1
                if len(topic_entities) > 2:
                    topic_more_than_2 += 1
                    topic_more_than_2_ids.append(qa.get("ID", "UNKNOWN_ID"))

            raw_answer = qa.get("answer", "")
            if isinstance(raw_answer, str):
                answer_count = 1 if raw_answer.strip() else 0
            elif isinstance(raw_answer, list):
                answer_count = len(raw_answer)
            else:
                answer_count = 0

            if answer_count == 1:
                answer_1 += 1
            elif answer_count > 1:
                answer_more += 1

            if len(topic_entities) > 1 and answer_count > 1:
                both_more += 1

            for mid, expected_label in topic_entities.items():
                total_entities += 1
                topic_check_tasks.append(executor.submit(
                    lambda m, l: (m, l, get_english_label(m)),
                    mid, expected_label
                ))

        for future in as_completed(topic_check_tasks):
            mid, expected, actual = future.result()
            if actual and actual.lower() == expected.lower():
                matched += 1
            else:
                failed.append({
                    "type": "topic_entity",
                    "mid": mid,
                    "expected": expected,
                    "actual": actual
                })

    print("🔍 Checking answer entity via SPARQL query...")
    answer_tasks = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for qa in qa_data:
            sparql_query = qa.get("sparql", "")
            annotated_answer = qa.get("answer", "").strip()
            qid = qa.get("ID")
            question = qa.get("question")

            def check_answer(qa):
                mids = run_sparql_query(qa.get("sparql", ""))
                answer_text = qa.get("answer", "").strip()
                for mid in mids:
                    label = get_english_label(mid)
                    if label and label.lower() == answer_text.lower():
                        qa["answer"] = {mid: label}
                        return qa, True
                qa["answer"] = {}  # No match
                return qa, False

            answer_tasks.append(executor.submit(check_answer, qa))

        for future in as_completed(answer_tasks):
            qa, matched_flag = future.result()
            updated_qa_data.append(qa)
            if matched_flag:
                answer_matched += 1
            else:
                answer_mismatch.append({
                    "type": "answer_entity",
                    "qid": qa.get("ID"),
                    "question": qa.get("question"),
                    "expected": qa.get("answer", ""),
                    "predicted_mid": None,
                    "predicted_label": None
                })
            answer_total += 1

    print("\n📊 Topic Entity Validation Summary")
    print(f"Total QA pairs: {total_qas}")
    print(f"Total Topic Entities Checked: {total_entities}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Failed: {len(failed)}")

    print("\n📈 Topic / Answer Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔻 Topic entity > 2 : {topic_more_than_2}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

    print("\n📊 Answer Entity Match Stats")
    print(f"Total SPARQL answer queries: {answer_total}")
    print(f"✅ Answer matched: {answer_matched}")
    print(f"❌ Answer mismatched: {len(answer_mismatch)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        json.dump(failed + answer_mismatch, fout, indent=2, ensure_ascii=False)
    print(f"\n📁 Mismatches saved to {OUTPUT_FILE}")

    with open("topic_entity_gt2_ids.txt", "w", encoding="utf-8") as f:
        for tid in topic_more_than_2_ids:
            f.write(tid + "\n")
    print(f"\n📁 Saved topic entity >2 IDs to topic_entity_gt2_ids.txt")

    with open(CWQ_NEW_FILE, "w", encoding="utf-8") as fout:
        json.dump(updated_qa_data, fout, indent=2, ensure_ascii=False)
    print(f"\n📁 Updated CWQ data with answer mids saved to {CWQ_NEW_FILE}")


if __name__ == "__main__":
    main()
