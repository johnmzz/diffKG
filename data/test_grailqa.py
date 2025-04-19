import json
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "grailqa.json"
SAVE_FAILED = True
FAILED_OUTPUT = "grailqa_failed_parallel.json"
FAILED_QID_LIST = "grailqa_all_answer_none_qids.json"

def fetch_label(mid):
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
        return bindings[0]["name"]["value"] if bindings else None
    except Exception:
        return None

def main():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_qas = len(data)

    topic_1 = topic_more = answer_1 = answer_more = both_more = 0

    match_tasks = []  # (qid, type, mid, expected_label)
    answer_mid_to_qid = {}  # mid → qid（仅限 answer）
    answer_mids_per_qid = {}  # qid → set(mid)

    for i, qa in enumerate(data):
        qid = str(qa.get("qid", f"QA-{i}"))
        topic_entities = qa.get("topic_entity", {})
        topic_count = len(topic_entities)

        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1

        for mid, expected in topic_entities.items():
            match_tasks.append((qid, "topic_entity", mid, expected))

        answer_entities = [a for a in qa.get("answer", []) if a.get("answer_type") == "Entity"]
        answer_count = len(answer_entities)

        if answer_count == 1:
            answer_1 += 1
        elif answer_count > 1:
            answer_more += 1

        if topic_count > 1 and answer_count > 1:
            both_more += 1

        for a in answer_entities:
            mid = a["answer_argument"]
            expected = a["entity_name"]
            match_tasks.append((qid, "answer", mid, expected))
            answer_mid_to_qid[mid] = qid
            answer_mids_per_qid.setdefault(qid, set()).add(mid)

        if i % 50 == 0:
            print(f"Queued {i}/{total_qas} examples...")

    total_entities = len(match_tasks)
    print(f"\n🚀 Launching parallel validation for {total_entities} MIDs...")

    matched = 0
    failed = []
    mismatch_none_label_count = 0
    answer_mids_with_none_label = set()

    failed_topic_count = 0
    failed_answer_count = 0

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(fetch_label, mid): (qid, mtype, mid, expected)
                   for (qid, mtype, mid, expected) in match_tasks}

        for i, future in enumerate(as_completed(futures)):
            qid, mtype, mid, expected = futures[future]
            actual = future.result()

            if actual and actual.lower() == expected.lower():
                matched += 1
            else:
                failed.append({
                    "qid": qid,
                    "type": mtype,
                    "mid": mid,
                    "expected": expected,
                    "actual": actual
                })
                if mtype == "answer":
                    if actual is None:
                        mismatch_none_label_count += 1
                        answer_mids_with_none_label.add(mid)
                    failed_answer_count += 1
                else:
                    failed_topic_count += 1

            if i % 200 == 0:
                print(f"Processed {i}/{len(futures)}...")

    # 📊 统计所有答案实体 label 全为 None 的 QID
    qids_all_answer_label_none = []
    for qid, mids in answer_mids_per_qid.items():
        if mids and mids.issubset(answer_mids_with_none_label):
            qids_all_answer_label_none.append(qid)

    # ✅ 输出统计
    print("\n📊 GrailQA MID Matching Summary")
    print(f"Total QA: {total_qas}")
    print(f"Total MID checked: {total_entities}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Failed: {len(failed)}")
    print(f"   ├── ❌ Topic entity mismatches: {failed_topic_count}")
    print(f"   └── ❌ Answer entity mismatches: {failed_answer_count}")

    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

    print("\n🆘 Label Extraction Issues")
    print(f"❌ Mismatches caused by missing KG label (label=None): {mismatch_none_label_count}")
    print(f"📎 Questions where ALL answers have label=None: {len(qids_all_answer_label_none)}")

    failed_topic = [f for f in failed if f["type"] == "topic_entity"]
    failed_answer = [f for f in failed if f["type"] == "answer"]

    if failed_topic:
        print("\n🔍 Failed Topic Entities (前几项):")
        for f in failed_topic[:10]:
            print(f"QID: {f['qid']} | MID: {f['mid']} | expected: '{f['expected']}' | actual: '{f['actual']}'")

    if failed_answer:
        print("\n🔍 Failed Answer Entities (前几项):")
        for f in failed_answer[:10]:
            print(f"QID: {f['qid']} | MID: {f['mid']} | expected: '{f['expected']}' | actual: '{f['actual']}'")

    if SAVE_FAILED:
        with open(FAILED_OUTPUT, "w", encoding="utf-8") as fout:
            json.dump(failed, fout, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved all failed entries to {FAILED_OUTPUT}")

        with open(FAILED_QID_LIST, "w", encoding="utf-8") as fq:
            json.dump(qids_all_answer_label_none, fq, indent=2)
        print(f"💾 Saved QIDs with all answer labels missing to {FAILED_QID_LIST}")

if __name__ == "__main__":
    main()
