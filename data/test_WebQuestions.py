import json
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "WebQuestions.json"
ANSWER_MATCH_OUTPUT = "webquestions_answer_match.json"
ANSWER_NO_MATCH_OUTPUT = "webquestions_answer_nomatch.json"

def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def check_answer_by_paged_neighbor(question, mid, answers, max_pages=1000, page_size=1000):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    seen = set()
    matches = []

    for direction in ['forward', 'backward']:
        for page in range(max_pages):
            offset = page * page_size
            if direction == 'forward':
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
            else:
                query = f"""
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?name ?p ?s
                FROM <http://freebase.com>
                WHERE {{
                    ?s ?p ns:{mid} .
                    ?s ns:type.object.name ?name .
                    FILTER (lang(?name) = "en")
                }}
                LIMIT {page_size} OFFSET {offset}
                """

            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            try:
                results = sparql.query().convert()
                bindings = results.get("results", {}).get("bindings", [])
                if not bindings:
                    break
                for b in bindings:
                    label = b["name"]["value"].strip()
                    relation = b["p"]["value"].replace("http://rdf.freebase.com/ns/", "")
                    entity_mid = None
                    if direction == 'forward':
                        entity_mid = b["o"]["value"].replace("http://rdf.freebase.com/ns/", "")
                    else:
                        entity_mid = b["s"]["value"].replace("http://rdf.freebase.com/ns/", "")
                    for ans in answers:
                        norm_key = (question.lower(), ans.lower(), direction)
                        if ans.lower() == label.lower() and norm_key not in seen:
                            seen.add(norm_key)
                            matches.append({
                                "matched_answer": ans,
                                "answer_mid": entity_mid,
                                "matched_label": label,
                                "matched_relation": relation,
                                "page": page + 1,
                                "direction": direction
                            })
            except Exception as e:
                print(f"⚠️ SPARQL {direction} error @ {mid} p{page}: {e}")
                break

    return matches


def main():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    existing_matches = load_existing(ANSWER_MATCH_OUTPUT)
    existing_nomatches = load_existing(ANSWER_NO_MATCH_OUTPUT)

    matched_qs = set(m["question"] for m in existing_matches)
    nomatched_qs = set(m["question"] for m in existing_nomatches)

    tasks = []
    for i, qa in enumerate(qa_data):
        question = qa.get("question", "").strip()
        if question in matched_qs:
            continue
        answers = qa.get("answers", [])
        topic_entities = qa.get("topic_entity", {})
        for mid in topic_entities:
            tasks.append((question, mid, answers))

    print(f"🔍 Remaining questions to process: {len(tasks)}")

    grouped_matches = []
    new_nomatches = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(check_answer_by_paged_neighbor, question, mid, answers): (question, mid, answers)
            for question, mid, answers in tasks
        }
        for i, future in enumerate(as_completed(futures)):
            question, mid, answers = futures[future]
            matches = future.result()
            if matches:
                grouped_matches.append({
                    "status": "match",
                    "question": question,
                    "topic_mid": mid,
                    "matched_answers": matches
                })
            else:
                new_nomatches.append({
                    "status": "no_match",
                    "question": question,
                    "topic_mid": mid,
                    "answers": answers
                })
            if i % 100 == 0:
                print(f"Progress: {i}/{len(futures)}")

    # Save
    all_matches = existing_matches + grouped_matches
    with open(ANSWER_MATCH_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2, ensure_ascii=False)

    still_nomatch_qs = {m["question"] for m in grouped_matches}
    final_nomatch = [n for n in existing_nomatches if n["question"] not in still_nomatch_qs] + new_nomatches
    with open(ANSWER_NO_MATCH_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_nomatch, f, indent=2, ensure_ascii=False)

    print(f"\n✅ New matched: {len(grouped_matches)} | ❌ Still no match: {len(new_nomatches)}")

if __name__ == "__main__":
    main()
