import json
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "WebQuestions.json"  # 替换成你的文件路径
SAVE_FAILED = True
FAILED_OUTPUT = "webquestions_failed.json"

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

    total_qas = len(qa_data)
    total_entities = 0
    matched = 0
    failed = []
    cache = {}

    # 新增：统计实体数量分布
    topic_1 = 0
    topic_more = 0
    answer_1 = 0
    answer_more = 0
    both_more = 0

    for i, qa in enumerate(qa_data):
        qid = qa.get("url", f"QA-{i}")
        topic_entities = qa.get("topic_entity", {})
        answers = qa.get("answers", [])

        topic_count = len(topic_entities)
        answer_count = len(answers)

        # 统计 topic entity 数量
        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1

        # 统计 answer 数量
        if answer_count == 1:
            answer_1 += 1
        elif answer_count > 1:
            answer_more += 1

        if topic_count > 1 and answer_count > 1:
            both_more += 1

        # 实体验证
        for mid, expected_label in topic_entities.items():
            total_entities += 1
            actual_label = get_english_label(sparql, mid, cache)
            if actual_label and actual_label.lower() == expected_label.lower():
                matched += 1
            else:
                failed.append({
                    "qid": qid,
                    "mid": mid,
                    "expected": expected_label,
                    "actual": actual_label
                })

        if i % 20 == 0:
            print(f"Progress: {i}/{total_qas}")

    # 输出统计结果
    print("\n📊 WebQuestions Topic Entity Validation Summary")
    print(f"Total QA pairs: {total_qas}")
    print(f"Total Topic Entities: {total_entities}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Failed: {len(failed)}")

    # 输出数量分布
    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

    # 输出失败样例并保存
    if failed:
        print("\n🔍 All Failed Mappings:")
        for f in failed[:10]:
            print(f"  QID: {f['qid']} | MID: {f['mid']} | expected: '{f['expected']}' | actual: '{f['actual']}'")

        if SAVE_FAILED:
            with open(FAILED_OUTPUT, "w", encoding="utf-8") as fout:
                json.dump(failed, fout, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved all failed entries to: {FAILED_OUTPUT}")

if __name__ == "__main__":
    main()
