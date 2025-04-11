import json
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "WebQSP.json"  # 替换为你的实际 WebQSP 文件路径

def get_english_label(sparql, mid, cache):
    if mid in cache:
        return cache[mid]

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
        label = bindings[0]["name"]["value"] if bindings else None
        cache[mid] = label
        return label
    except Exception as e:
        print(f"⚠️ SPARQL error for MID {mid}: {e}")
        cache[mid] = None
        return None

def main():
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    cache = {}
    total_qas = len(qa_data)

    total_topic_entities = 0
    matched_topic = 0
    topic_failed = []

    total_answer_entities = 0
    matched_answer = 0
    answer_failed = []

    # 统计：topic / answer entity 数量级别
    topic_1 = 0
    topic_more = 0
    answer_1 = 0
    answer_more = 0
    both_more = 0

    for i, qa in enumerate(qa_data):
        qid = qa.get("QuestionId", f"QA-{i}")
        topic_entities = qa.get("topic_entity", {})
        topic_count = len(topic_entities)

        # ✅ topic entity 数量统计
        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1

        # ✅ 实体验证：topic_entity
        for mid, expected_label in topic_entities.items():
            total_topic_entities += 1
            actual_label = get_english_label(sparql, mid, cache)
            if actual_label and actual_label.lower() == expected_label.lower():
                matched_topic += 1
            else:
                topic_failed.append({
                    "qid": qid,
                    "mid": mid,
                    "expected": expected_label,
                    "actual": actual_label
                })

        # ✅ answer entity 数量合并统计
        answer_entity_set = set()
        for parse in qa.get("Parses", []):
            for answer in parse.get("Answers", []):
                if answer["AnswerType"] == "Entity":
                    mid = answer["AnswerArgument"]
                    expected_label = answer["EntityName"]
                    answer_entity_set.add(mid)

                    # 实体验证：答案实体
                    total_answer_entities += 1
                    actual_label = get_english_label(sparql, mid, cache)
                    if actual_label and actual_label.lower() == expected_label.lower():
                        matched_answer += 1
                    else:
                        answer_failed.append({
                            "qid": qid,
                            "mid": mid,
                            "expected": expected_label,
                            "actual": actual_label
                        })

        answer_count = len(answer_entity_set)
        if answer_count == 1:
            answer_1 += 1
        elif answer_count > 1:
            answer_more += 1

        if topic_count > 1 and answer_count > 1:
            both_more += 1

        if i % 20 == 0:
            print(f"Progress: {i}/{total_qas}")

    # ✅ 输出验证结果
    print("\n📊 WebQSP Entity Validation Summary")
    print(f"Total QA pairs: {total_qas}")
    print(f"Total Topic Entities: {total_topic_entities} | ✅ Matched: {matched_topic} | ❌ Failed: {len(topic_failed)}")
    print(f"Total Answer Entities: {total_answer_entities} | ✅ Matched: {matched_answer} | ❌ Failed: {len(answer_failed)}")

    # ✅ 输出实体数量统计
    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

    if topic_failed or answer_failed:
        print("\n🔍 Sample Failures:")
        for f in (topic_failed + answer_failed)[:10]:
            print(f"  QID: {f['qid']} | MID: {f['mid']} | expected: '{f['expected']}' | actual: '{f['actual']}'")

if __name__ == "__main__":
    main()
