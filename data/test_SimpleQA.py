import json
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "SimpleQA.json"  # 替换成你的实际文件路径

def get_english_label(sparql, mid):
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
        else:
            return None
    except Exception as e:
        print(f"⚠️ SPARQL error for MID {mid}: {e}")
        return None

def main():
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    total = len(qa_data)
    matched = 0
    failed = []

    # 统计计数器
    topic_1 = 0
    topic_more = 0
    answer_1 = 0
    answer_more = 0
    both_more = 0

    for i, qa in enumerate(qa_data):
        topic_entities = qa.get("topic_entity", {})
        topic_count = len(topic_entities)

        # Topic entity 统计
        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1

        # Answer 可能是 str 或 list
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

        if topic_count > 1 and answer_count > 1:
            both_more += 1

        # 实体验证
        for mid, expected_label in topic_entities.items():
            label = get_english_label(sparql, mid)
            if label and label.lower() == expected_label.lower():
                matched += 1
            else:
                failed.append({
                    "mid": mid,
                    "expected": expected_label,
                    "actual": label
                })

        if i % 20 == 0:
            print(f"Progress: {i}/{total}")

    # 打印统计结果
    print("\n📊 SimpleQA Validation Summary")
    print(f"Total QA pairs: {total}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Failed: {len(failed)}")

    if failed:
        print("\n🔍 Some failures:")
        for f in failed[:5]:
            print(f"  MID: {f['mid']}, expected: '{f['expected']}', actual: '{f['actual']}'")

    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

if __name__ == "__main__":
    main()
