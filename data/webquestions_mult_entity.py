import json
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
INPUT_FILE = "webquestions_topic_more_than_one.json"
OUTPUT_FILE = "webquestions_source_added.json"

def check_relation(from_mid, to_mid, relation):
    """
    Returns True if from_mid --[relation]--> to_mid exists in KG
    """
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    ASK {{
        ns:{from_mid} ns:{relation} ns:{to_mid} .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        return sparql.query().convert().get("boolean", False)
    except Exception as e:
        print(f"⚠️ SPARQL query error: {e}")
        return False

def annotate_sources(data):
    for item in data:
        topic_entities = item.get("topic_entity", {})  # mid → label
        real_answers = item.get("real_answers", {})    # mid → {label, relation}

        for ans_mid, ans_info in real_answers.items():
            relation = ans_info.get("relation")
            found = False

            for topic_mid, topic_label in topic_entities.items():
                if check_relation(topic_mid, ans_mid, relation):
                    ans_info["source"] = {
                        "mid": topic_mid,
                        "label": topic_label
                    }
                    found = True
                    break

            if not found:
                ans_info["source"] = {
                    "mid": "NOT_FOUND",
                    "label": "NOT_FOUND"
                }

    return data

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔍 Processing {len(data)} QA entries...")
    annotated = annotate_sources(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)
    print(f"✅ Done. Annotated file saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
