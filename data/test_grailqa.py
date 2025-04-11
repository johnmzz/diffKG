import json
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "grailqa.json"  # 替换为你的实际文件路径
SAVE_FAILED = True
FAILED_OUTPUT = "grailqa_failed.json"

def get_label(sparql, mid, cache):
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
        data = json.load(f)

    total_qas = len(data)
    total_entities = 0
    matched = 0
    failed = []
    cache = {}

    # 新增统计变量
    topic_1 = 0
    topic_more = 0
    answer_1 = 0
    answer_more = 0
    both_more = 0

    for i, qa in enumerate(data):
        qid = qa.get("qid", f"QA-{i}")

        # === Topic Entity ===
        topic_entities = qa.get("topic_entity", {})
        topic_count = len(topic_entities)
        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1

        for mid, expected in topic_entities.items():
            total_entities += 1
            actual = get_label(sparql, mid, cache)
            if actual and actual.lower() == expected.lower():
                matched += 1
            else:
                failed.append({
                    "qid": qid,
                    "type": "topic_entity",
                    "mid": mid,
                    "expected": expected,
                    "actual": actual
                })

        # === Answer Entity ===
        answer_entities = [a for a in qa.get("answer", []) if a.get("answer_type") == "Entity"]
        answer_count = len(answer_entities)
        if answer_count == 1:
            answer_1 += 1
        elif answer_count > 1:
            answer_more += 1

        if topic_count > 1 and answer_count > 1:
            both_more += 1

        for answer in answer_entities:
            mid = answer["answer_argument"]
            expected = answer["entity_name"]
            total_entities += 1
            actual = get_label(sparql, mid, cache)
            if actual and actual.lower() == expected.lower():
                matched += 1
            else:
                failed.append({
                    "qid": qid,
                    "type": "answer",
                    "mid": mid,
                    "expected": expected,
                    "actual": actual
                })

        if i % 20 == 0:
            print(f"Progress: {i}/{total_qas}")

    # === 输出结果 ===
    print("\n📊 GrailQA MID Matching Summary")
    print(f"Total QA: {total_qas}")
    print(f"Total MID checked: {total_entities}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Failed: {len(failed)}")

    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

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

if __name__ == "__main__":
    main()

