import json
import re
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
INPUT_FILE = "WebQSP.json"
OUTPUT_FILE = "WebQSP_new.json"

def fetch_label(mid):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?name
    FROM <http://freebase.com>
    WHERE {{
        ns:{mid} ns:type.object.name ?name .
        FILTER(lang(?name) = "en")
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        return bindings[0]["name"]["value"] if bindings else None
    except:
        return None

def fetch_type(mid):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?type
    FROM <http://freebase.com>
    WHERE {{
        ns:{mid} ns:type.object.type ?type .
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            return bindings[0]["type"]["value"].replace("http://rdf.freebase.com/ns/", "")
    except:
        return None

def extract_mids_from_sparql(sparql_text):
    return re.findall(r'ns:(m\.[a-zA-Z0-9_]+)', sparql_text)

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        question = item.get("RawQuestion", "N/A")
        topic_entity_dict = item.get("topic_entity", {})
        mids_in_topic = set(topic_entity_dict.keys())

        for parse in item.get("Parses", []):
            sparql = parse.get("Sparql", "")
            triples = [l for l in sparql.split('\n') if 'ns:' in l and not l.strip().startswith('#')]
            parse["Triples"] = triples  # optional: add for debug

            print(f"\n🟢 Question: {question}")
            for line in triples:
                print(f"   ➤ {line.strip()}")

            # extract all m.xxxxxx used in query
            mids_in_sparql = set(extract_mids_from_sparql(sparql))
            for mid in mids_in_sparql:
                if mid not in topic_entity_dict:
                    label = fetch_label(mid)
                    if not label:
                        type_info = fetch_type(mid)
                        if type_info:
                            label = f"UNKNOWN, type = {type_info}"
                        else:
                            label = "UNKNOWN"
                    topic_entity_dict[mid] = label
                    print(f"     └ added: {mid} ({label})")
                else:
                    print(f"     └ exists: {mid} ({topic_entity_dict[mid]})")

        # update topic_entity
        item["topic_entity"] = topic_entity_dict

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)
    print(f"\n✅ Updated file saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
