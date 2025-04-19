import json
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import difflib

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "WebQuestions.json"
FAILED_OUTPUT = "webquestions_failed.json"
UNKNOWN_OUTPUT = "webquestions_unknownmid_qids.json"
ALL_NONE_OUTPUT = "webquestions_all_topic_none_qids.json"
ANSWER_MATCH_OUTPUT = "webquestions_answer_match.json"
ANSWER_NO_MATCH_OUTPUT = "webquestions_answer_nomatch.json"
SAVE_FAILED = True

# 加载已有的匹配记录（用于跳过）
def load_existing_answers(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 判断答案是否相近（宽松匹配）
def fuzzy_match(ans, label):
    ans, label = ans.lower().strip(), label.lower().strip()
    if ans in label or label in ans:
        return True, True, 1.0
    ratio = difflib.SequenceMatcher(None, ans, label).ratio()
    return ratio > 0.85, ratio > 0.85, ratio

# 分页查询邻居并找答案
def check_answer_by_paged_neighbor(qid, mid, question, answers, page_size=300, max_pages=20):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    candidates = []
    for page in range(max_pages):
        offset = page * page_size
        query = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?name ?p ?o
        FROM <http://freebase.com>
        WHERE {{
            ns:{mid} ?p ?o .
            ?o ns:type.object.name ?name .
            FILTER (lang(?name) = "en")
        }}
        LIMIT {page_size} OFFSET {offset}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            for b in bindings:
                label = b["name"]["value"].strip()
                relation = b["p"]["value"].replace("http://rdf.freebase.com/ns/", "")
                answer_mid = b["o"]["value"].replace("http://rdf.freebase.com/ns/", "")
                for ans in answers:
                    matched, is_fuzzy, score = fuzzy_match(ans, label)
                    if matched:
                        candidates.append({
                            "qid": qid,
                            "question": question,
                            "topic_mid": mid,
                            "matched_answer": ans,
                            "answer_mid": answer_mid,
                            "matched_label": label,
                            "matched_relation": relation,
                            "page": page + 1,
                            "fuzzy": is_fuzzy,
                            "score": score
                        })
        except Exception as e:
            print(f"⚠️ SPARQL error for {mid} page {page}: {e}")
            break

    if candidates:
        best = max(candidates, key=lambda x: x["score"])
        best["status"] = "match"
        return best

    return {
        "status": "no_match",
        "qid": qid,
        "question": question,
        "topic_mid": mid,
        "answers": answers
    }


def loop_until_stable(max_rounds=20, sleep_time=3):
    prev_count = -1
    for round_num in range(1, max_rounds + 1):
        print(f"\n🚀 Iteration {round_num}: Checking for new matches...")
        main()
        print(f"🕒 Sleeping {sleep_time}s before next round...")
        time.sleep(sleep_time)


def main():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    existing_matches = load_existing_answers(ANSWER_MATCH_OUTPUT)
    already_matched_qids = set(item["qid"] for item in existing_matches)

    existing_nomatches = load_existing_answers(ANSWER_NO_MATCH_OUTPUT)
    nomatch_qid_set = set(item["qid"] for item in existing_nomatches)

    new_matches = []
    new_nomatches = []
    topic_tasks = []

    for i, qa in enumerate(qa_data):
        qid = qa.get("url", f"QA-{i}")
        if qid in already_matched_qids:
            continue
        question = qa.get("question", "")
        topic_entities = qa.get("topic_entity", {})
        answers = qa.get("answers", [])
        for mid in topic_entities:
            topic_tasks.append((qid, mid, question, answers))

    print(f"📦 Starting answer neighbor search for {len(topic_tasks)} examples...")

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(check_answer_by_paged_neighbor, qid, mid, question, answers): qid
            for qid, mid, question, answers in topic_tasks
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result["status"] == "match":
                new_matches.append(result)
                nomatch_qid_set.discard(result["qid"])
            else:
                new_nomatches.append(result)
            if i % 100 == 0:
                print(f"🔄 Checked {i}/{len(futures)}")

    fuzzy_matches = sum(1 for r in new_matches if r.get("fuzzy"))
    exact_matches = len(new_matches) - fuzzy_matches

    print(f"\n✅ Newly matched answers: {len(new_matches)}")
    print(f"   🔹 Exact matches: {exact_matches}")
    print(f"   🔸 Fuzzy matches: {fuzzy_matches}")
    print(f"❌ Still no match: {len(new_nomatches)}")

    # 合并保存
    all_matches = existing_matches + new_matches
    with open(ANSWER_MATCH_OUTPUT, "w", encoding="utf-8") as fout:
        json.dump(all_matches, fout, indent=2, ensure_ascii=False)

    filtered_nomatch = [entry for entry in existing_nomatches if entry["qid"] in nomatch_qid_set] + new_nomatches
    with open(ANSWER_NO_MATCH_OUTPUT, "w", encoding="utf-8") as fout:
        json.dump(filtered_nomatch, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    loop_until_stable()
