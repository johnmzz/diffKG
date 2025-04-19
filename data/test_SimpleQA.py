import json
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "SimpleQA.json"

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

def check_answer_by_local_relation(topic_mid, relation_keyword, answer_text):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    # 1. 获取 topic_mid 所有 predicate
    query1 = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?p
    FROM <http://freebase.com>
    WHERE {{
        ns:{topic_mid} ?p ?o .
    }}
    LIMIT 200
    """
    sparql.setQuery(query1)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        predicates = [
            b["p"]["value"].replace("http://rdf.freebase.com/ns/", "")
            for b in results["results"]["bindings"]
        ]
    except Exception as e:
        print(f"⚠️ Failed to fetch relations for {topic_mid}")
        return False, [], {}

    # 2. 模糊匹配包含 keyword 的 predicate
    matched_preds = [p for p in predicates if relation_keyword.lower() in p.lower().split(".")[-1]]
    tried_answers = {}

    if not matched_preds:
        return False, [], tried_answers

    # 3. 查询这些 predicate 的对象 label
    for predicate in matched_preds:
        query2 = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?name
        FROM <http://freebase.com>
        WHERE {{
            ns:{topic_mid} ns:{predicate} ?o .
            ?o ns:type.object.name ?name .
            FILTER (lang(?name) = "en")
        }}
        LIMIT 200
        """
        sparql.setQuery(query2)
        sparql.setReturnFormat(JSON)
        try:
            res = sparql.query().convert()
            neighbor_names = [
                b["name"]["value"].strip()
                for b in res["results"]["bindings"]
                if "name" in b
            ]
            tried_answers[predicate] = neighbor_names

            if answer_text.lower() in (n.lower() for n in neighbor_names):
                return True, [predicate], tried_answers  # ✅ 匹配成功
        except Exception:
            continue

    return False, matched_preds, tried_answers



def main():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    total_qas = len(qa_data)
    topic_1 = topic_more = answer_1 = answer_more = both_more = 0
    answer_match = 0
    answer_nomatch = 0
    relation_check_failures = []
    relation_check_successes = []

    check_tasks = []  # (mid, expected_label)
    qid_map = []      # 保存 (qid, mid) 用于回溯

    for i, qa in enumerate(qa_data):
        topic_entities = qa.get("topic_entity", {})
        topic_count = len(topic_entities)

        if topic_count == 1:
            topic_1 += 1
        elif topic_count > 1:
            topic_more += 1

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

        # 🧠 新功能：检查 answer 是否在 topic_entity + relation 的邻居中
        if topic_count == 1 and answer_count == 1 and "relation" in qa:
            topic_mid = next(iter(topic_entities))
            relation = qa["relation"]
            answer_text = raw_answer.strip().lower()

            match_found, tried_predicates, tried_answers = check_answer_by_local_relation(topic_mid, relation, answer_text)

            if match_found:
                answer_match += 1
                relation_check_successes.append({
                    "id": qa.get("id", f"QA-{i}"),
                    "question": qa.get("question", ""),
                    "topic_entity": topic_entities,
                    "relation": relation,
                    "expected_answer": raw_answer,
                    "matched_predicate": tried_predicates[0],
                    "all_labels": tried_answers.get(tried_predicates[0], [])
                })
            else:
                answer_nomatch += 1
                relation_check_failures.append({
                    "id": qa.get("id", f"QA-{i}"),
                    "question": qa.get("question", ""),
                    "topic_entity": topic_entities,
                    "relation": relation,
                    "expected_answer": raw_answer,
                    "tried_predicates": tried_predicates[:5],
                    "tried_answers": {k: v[:10] for k, v in tried_answers.items()}
                })

        # 原始实体检查任务
        for mid, expected in topic_entities.items():
            check_tasks.append((mid, expected))
            qid_map.append(qa.get("id", f"QA-{i}"))

        if i % 50 == 0:
            print(f"Prepared {i}/{total_qas} QA pairs...")

    print(f"\n🚀 Launching parallel SPARQL validation for {len(check_tasks)} topic entities...")

    matched = 0
    failed = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(fetch_label, mid): (i, mid, expected)
                   for i, (mid, expected) in enumerate(check_tasks)}

        for i, future in enumerate(as_completed(futures)):
            idx, mid, expected = futures[future]
            actual = future.result()
            if actual and actual.lower() == expected.lower():
                matched += 1
            else:
                failed.append({
                    "qid": qid_map[idx],
                    "mid": mid,
                    "expected": expected,
                    "actual": actual
                })
            if i % 200 == 0:
                print(f"Validated {i}/{len(futures)} entities...")

    # === 输出统计 ===
    print("\n📊 SimpleQA Entity Validation Summary")
    print(f"Total QA pairs: {total_qas}")
    print(f"Total topic entities checked: {len(check_tasks)}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Failed: {len(failed)}")

    if failed:
        print("\n🔍 Sample Failures:")
        for f in failed[:5]:
            print(f"  QID: {f['qid']} | MID: {f['mid']} | expected: '{f['expected']}' | actual: '{f['actual']}'")

    print("\n📈 Topic / Answer Entity Count Stats")
    print(f"🔹 Topic entity = 1 : {topic_1}")
    print(f"🔸 Topic entity > 1 : {topic_more}")
    print(f"🔹 Answer entity = 1 : {answer_1}")
    print(f"🔸 Answer entity > 1 : {answer_more}")
    print(f"⚠️  Both topic & answer > 1 : {both_more}")

    print("\n🔗 Relation-based Answer Check (from neighbors):")
    print(f"✅ Matched in neighbors: {answer_match}")
    print(f"❌ Not found in neighbors: {answer_nomatch}")

    with open("simpleqa_failed.json", "w", encoding="utf-8") as fout:
        json.dump(failed, fout, indent=2)

    with open("relation_check_failed.json", "w", encoding="utf-8") as fout:
        json.dump(relation_check_failures, fout, indent=2)

    with open("relation_check_success.json", "w", encoding="utf-8") as fout:
        json.dump(relation_check_successes, fout, indent=2)

if __name__ == "__main__":
    main()
