import json
import os
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "WebQuestions.json"

FAILED_OUTPUT = "webquestions_failed.json"
UNKNOWN_OUTPUT = "webquestions_unknownmid_qids.json"
ALL_NONE_OUTPUT = "webquestions_all_topic_none_qids.json"
TOPIC_MORE_OUTPUT = "webquestions_topic_more_than_one.json"

SAVE_FAILED = True

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
        qa_data = json.load(f)

    total_qas = len(qa_data)
    topic_1 = topic_more = answer_1 = answer_more = both_more = 0

    topic_tasks = []
    qid_to_all_topic_mids = {}
    unknown_mid_qids = []
    qid_to_question = {}
    topic_more_qas = []

    for i, qa in enumerate(qa_data):
        qid = qa.get("url", f"QA-{i}")
        question_text = qa.get("question", "")
        topic_entities = qa.get("topic_entity", {})
        answers = qa.get("answers", [])

        qid_to_question[qid] = question_text
        topic_count = len(topic_entities)
        answer_count = len(answers)

        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1
            topic_more_qas.append(qa)

        if answer_count == 1:
            answer_1 += 1
        elif answer_count > 1:
            answer_more += 1

        if topic_count > 1 and answer_count > 1:
            both_more += 1

        qid_to_all_topic_mids[qid] = set(topic_entities.keys())
        for mid, expected in topic_entities.items():
            if mid.lower() == "unknownmid":
                unknown_mid_qids.append(qid)
                continue
            topic_tasks.append((qid, mid, expected, question_text))

        if i % 50 == 0:
            print(f"Prepared {i}/{total_qas} QA entries...")

    print(f"\n🚀 Launching SPARQL validation for {len(topic_tasks)} topic entities (excluding UnknownMID)...")

    matched = 0
    failed = []
    none_label_mids = set()
    qid_to_failed_mids = {}

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(fetch_label, mid): (qid, mid, expected, question)
            for (qid, mid, expected, question) in topic_tasks
        }

        for i, future in enumerate(as_completed(futures)):
            qid, mid, expected, question = futures[future]
            actual = future.result()
            if actual and actual.lower() == expected.lower():
                matched += 1
            else:
                failed.append({
                    "qid": qid,
                    "question": question,
                    "mid": mid,
                    "expected": expected,
                    "actual": actual
                })
                if actual is None:
                    none_label_mids.add(mid)
                if qid not in qid_to_failed_mids:
                    qid_to_failed_mids[qid] = set()
                qid_to_failed_mids[qid].add(mid)

            if i % 200 == 0:
                print(f"Validated {i}/{len(futures)}...")

    # 检查所有 topic 都为 None 的问题
    qids_all_none = []
    for qid, all_mids in qid_to_all_topic_mids.items():
        if all(mid in none_label_mids for mid in all_mids if mid != "UnknownMID"):
            qids_all_none.append(qid)

    # === 输出统计 ===
    print("\n📊 WebQuestions Topic Entity Validation Summary")
    print(f"Total QA pairs: {total_qas}")
    print(f"Total Topic Entities Checked: {len(topic_tasks)}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Failed: {len(failed)}")
    print(f"❌ Failed with label=None: {len(none_label_mids)}")
    print(f"⚠️  Questions where all topic entity labels are None: {len(qids_all_none)}")
    print(f"⚠️  Questions with UnknownMID: {len(set(unknown_mid_qids))}")

    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

    if failed:
        print("\n🔍 Sample Failures:")
        for f in failed[:10]:
            print(f"  QID: {f['qid']} | Q: {f['question']}")
            print(f"     MID: {f['mid']} | expected: '{f['expected']}' | actual: '{f['actual']}'")

    # === 保存结果 ===
    if SAVE_FAILED:
        with open(FAILED_OUTPUT, "w", encoding="utf-8") as fout:
            json.dump(failed, fout, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved all failed entries to: {FAILED_OUTPUT}")

        with open(ALL_NONE_OUTPUT, "w", encoding="utf-8") as fout:
            json.dump(qids_all_none, fout, indent=2)
            print(f"💾 Saved QIDs with all topic labels None to: {ALL_NONE_OUTPUT}")

        with open(UNKNOWN_OUTPUT, "w", encoding="utf-8") as fout:
            json.dump(list(set(unknown_mid_qids)), fout, indent=2)
            print(f"💾 Saved QIDs with UnknownMID to: {UNKNOWN_OUTPUT}")

        with open(TOPIC_MORE_OUTPUT, "w", encoding="utf-8") as fout:
            json.dump(topic_more_qas, fout, indent=2, ensure_ascii=False)
            print(f"💾 Saved all topic_entity > 1 QAs to: {TOPIC_MORE_OUTPUT}")

if __name__ == "__main__":
    main()
