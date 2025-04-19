import json
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "WebQSP.json"
OUTPUT_FILE = "mismatch_report_webqsp_parallel_all.json"

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
    except:
        return None
    return None

def main():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    total_qas = len(qa_data)

    topic_1 = topic_more = answer_1 = answer_more = both_more = 0

    topic_tasks = []
    answer_tasks = []

    # ⏱️ answer mid 到 question ID 映射（用于“所有答案都为 None”的统计）
    mid_to_qid = {}

    for i, qa in enumerate(qa_data):
        qid = qa.get("QuestionId", f"QA-{i}")
        topic_entities = qa.get("topic_entity", {})
        topic_count = len(topic_entities)

        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1

        answer_entity_set = set()
        answer_pairs = []

        for parse in qa.get("Parses", []):
            for ans in parse.get("Answers", []):
                if ans["AnswerType"] == "Entity":
                    mid = ans["AnswerArgument"]
                    label = ans["EntityName"]
                    answer_entity_set.add(mid)
                    answer_pairs.append((qid, mid, label))
                    mid_to_qid[mid] = qid  # 建立 mid ↔ qid 映射

        answer_count = len(answer_entity_set)
        if answer_count == 1:
            answer_1 += 1
        elif answer_count > 1:
            answer_more += 1
        if topic_count > 1 and answer_count > 1:
            both_more += 1

        for mid, label in topic_entities.items():
            topic_tasks.append(("topic_entity", qid, mid, label))

        for (qid, mid, label) in answer_pairs:
            answer_tasks.append(("answer_entity", qid, mid, label))

        if i % 50 == 0:
            print(f"Queued {i}/{total_qas} examples...")

    total_topic_entities = len(topic_tasks)
    total_answer_entities = len(answer_tasks)

    print(f"\n🚀 Launching parallel entity validation: {total_topic_entities} topic, {total_answer_entities} answer")

    mismatches = []
    matched_topic = matched_answer = 0
    answer_none_label_qids = set()
    mismatch_none_label_count = 0

    # 用于统计哪些 answer 的 mid 对应 KG 查不到 label
    answer_mid_none_label = set()

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(get_english_label, mid): (etype, qid, mid, expected)
                   for (etype, qid, mid, expected) in topic_tasks + answer_tasks}

        for i, future in enumerate(as_completed(futures)):
            etype, qid, mid, expected = futures[future]
            actual = future.result()
            if actual and actual.lower() == expected.lower():
                if etype == "topic_entity":
                    matched_topic += 1
                else:
                    matched_answer += 1
            else:
                mismatches.append({
                    "type": etype,
                    "qid": qid,
                    "mid": mid,
                    "expected": expected,
                    "actual": actual
                })
                if etype == "answer_entity" and actual is None:
                    mismatch_none_label_count += 1
                    answer_mid_none_label.add(mid)

            if i % 200 == 0:
                print(f"Processed {i}/{len(futures)}...")

    # 🚨 统计 “所有答案都查不到 label 的问题”
    qid_to_failed_mids = {}
    for mid in answer_mid_none_label:
        qid = mid_to_qid[mid]
        qid_to_failed_mids.setdefault(qid, []).append(mid)

    # 将每个问题的所有 answer mid 拿出来对比
    qids_all_none = 0
    for qa in qa_data:
        qid = qa.get("QuestionId")
        answer_mids = set()
        for parse in qa.get("Parses", []):
            for ans in parse.get("Answers", []):
                if ans["AnswerType"] == "Entity":
                    answer_mids.add(ans["AnswerArgument"])
        if answer_mids and answer_mids.issubset(answer_mid_none_label):
            qids_all_none += 1

    # 输出统计结果
    print("\n📊 WebQSP Entity Validation Summary")
    print(f"Total QA pairs: {total_qas}")
    print(f"Total Topic Entities: {total_topic_entities} | ✅ Matched: {matched_topic} | ❌ Failed: {total_topic_entities - matched_topic}")
    print(f"Total Answer Entities: {total_answer_entities} | ✅ Matched: {matched_answer} | ❌ Failed: {total_answer_entities - matched_answer}")

    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

    print("\n🆘 Label Extraction Issues")
    print(f"❌ Mismatches caused by missing KG label (label=None): {mismatch_none_label_count}")
    print(f"📎 Questions where ALL answers have label=None: {qids_all_none}")

    if mismatches:
        print("\n🔍 Sample Mismatches:")
        for f in mismatches[:10]:
            print(f"  QID: {f['qid']} | MID: {f['mid']} | expected: '{f['expected']}' | actual: '{f['actual']}'")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        json.dump(mismatches, fout, indent=2, ensure_ascii=False)
    print(f"\n📁 Mismatches saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
