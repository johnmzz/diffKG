import json
import os
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
QA_FILE = "SimpleQA.json"
ANSWER_MATCH_OUTPUT = "simpleqa_answer_match.json"
ANSWER_NO_MATCH_OUTPUT = "simpleqa_answer_nomatch.json"

FUZZY_THRESHOLD = 0.75
FUZZY_TOP_K = 10


def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def check_answer_by_paged_neighbor(question, mid, answers, max_pages=1000, page_size=1000):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    seen = set()
    exact_matches = []
    fuzzy_candidates = []

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
            if not bindings:
                break
            for b in bindings:
                label = b["name"]["value"].strip()
                relation = b["p"]["value"].replace("http://rdf.freebase.com/ns/", "")
                answer_mid = b["o"]["value"].replace("http://rdf.freebase.com/ns/", "")
                for ans in answers:
                    norm_key = (question.lower(), ans.lower())
                    if ans.lower() == label.lower() and norm_key not in seen:
                        seen.add(norm_key)
                        exact_matches.append({
                            "matched_answer": ans,
                            "answer_mid": answer_mid,
                            "matched_label": label,
                            "matched_relation": relation,
                            "page": page + 1,
                            "match_type": "exact"
                        })
                    else:
                        score = similarity(ans, label)
                        if score > FUZZY_THRESHOLD:
                            fuzzy_candidates.append({
                                "matched_answer": ans,
                                "answer_mid": answer_mid,
                                "matched_label": label,
                                "matched_relation": relation,
                                "page": page + 1,
                                "match_type": "fuzzy",
                                "score": round(score, 4)
                            })
        except Exception as e:
            print(f"⚠️ SPARQL error @ {mid} p{page}: {e}")
            break

    if exact_matches:
        return exact_matches
    return sorted(fuzzy_candidates, key=lambda x: x["score"], reverse=True)[:FUZZY_TOP_K]


def check_2hop_via_compound_type(topic_mid, gold_answers):
    compound_mids = []
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query1 = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?neighbor
    FROM <http://freebase.com>
    WHERE {{
        ns:{topic_mid} ?p ?neighbor .
        ?neighbor ns:type.object.type ?type .
    }}
    LIMIT 1000
    """
    sparql.setQuery(query1)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        for b in results["results"]["bindings"]:
            n = b["neighbor"]["value"]
            if n.startswith("http://rdf.freebase.com/ns/"):
                mid = n.replace("http://rdf.freebase.com/ns/", "")
                compound_mids.append(mid)
    except Exception as e:
        print(f"❌ Failed to fetch compound nodes for {topic_mid}: {e}")
        return []

    matches = []
    for compound_mid in compound_mids:
        sparql = SPARQLWrapper(SPARQL_ENDPOINT)
        query2 = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?name ?p ?o
        FROM <http://freebase.com>
        WHERE {{
            ns:{compound_mid} ?p ?o .
            ?o ns:type.object.name ?name .
            FILTER (lang(?name) = "en")
        }}
        LIMIT 500
        """
        sparql.setQuery(query2)
        sparql.setReturnFormat(JSON)
        try:
            res = sparql.query().convert()
            for b in res["results"]["bindings"]:
                label = b["name"]["value"].strip()
                pred = b["p"]["value"].replace("http://rdf.freebase.com/ns/", "")
                obj_mid = b["o"]["value"].replace("http://rdf.freebase.com/ns/", "")
                for gold in gold_answers:
                    if gold.lower() == label.lower():
                        matches.append({
                            "compound_mid": compound_mid,
                            "matched_label": label,
                            "answer_mid": obj_mid,
                            "relation": pred,
                            "via": "2hop-compound"
                        })
        except Exception as e:
            print(f"⚠️ Failed 2-hop query from {compound_mid}: {e}")
            continue

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
        answer = qa.get("answer", "").strip()
        topic_entities = qa.get("topic_entity", {})
        for mid in topic_entities:
            tasks.append((question, mid, [answer]))

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
            if not matches:
                matches = check_2hop_via_compound_type(mid, answers)

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
            if i % 50 == 0:
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
