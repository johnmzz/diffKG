import json
import re
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
GRAILQA_FILE = "grailqa.json"  # ← 修改为你的实际路径

def get_label_for_mids(mids):
    if not mids:
        return {}
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    values_clause = " ".join([f"ns:{mid}" for mid in mids])
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?mid ?name
    FROM <http://freebase.com>
    WHERE {{
      VALUES ?mid {{ {values_clause} }}
      ?mid ns:type.object.name ?name .
      FILTER (lang(?name) = "en")
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        labels = {}
        for b in results["results"]["bindings"]:
            mid = b["mid"]["value"].replace("http://rdf.freebase.com/ns/", "")
            label = b["name"]["value"]
            labels[mid] = label
        return labels
    except Exception as e:
        print("SPARQL label fetch failed:", e)
        return {}

def parse_graph_query(graph_query):
    nid_to_node = {}
    triples = []
    all_mids = set()

    for node in graph_query['nodes']:
        nid = node['nid']
        node_id = node['id']
        name = node.get('friendly_name', node_id)
        nid_to_node[nid] = (node_id, name)
        if node_id.startswith('m.'):
            all_mids.add(node_id)

    for edge in graph_query['edges']:
        src = edge['start']
        dst = edge['end']
        rel = edge['relation']
        rel_name = edge.get('friendly_name', rel)
        h_id, h_name = nid_to_node[src]
        t_id, t_name = nid_to_node[dst]
        if t_id.startswith('m.'):
            all_mids.add(t_id)
        triples.append((h_id, rel, t_id, h_name, rel_name, t_name))

    return triples, all_mids

def display_question_graph(qid, question, graph_query, answers):
    triples, all_mids = parse_graph_query(graph_query)
    label_map = get_label_for_mids(all_mids)
    print(f"\n🟢 QID: {qid}")
    print(f"🟢 Question: {question}")
    for h_id, rel, t_id, h_name, rel_name, t_name in triples:
        h_label = label_map.get(h_id, h_name) if h_id.startswith('m.') else h_name
        t_label = label_map.get(t_id, t_name) if t_id.startswith('m.') else t_name
        h_disp = f"{h_id} ({h_label})" if h_id.startswith('m.') else h_label
        t_disp = f"{t_id} ({t_label})" if t_id.startswith('m.') else t_label
        print(f"   ➤ {h_disp} --{rel}--> {t_disp}")
    for ans in answers:
        if ans["answer_type"] == "Entity":
            mid = ans["answer_argument"]
            label = ans["entity_name"]
            print(f"        └ Answer: {mid} ({label})")
    return len(triples)

def process_grailqa(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hop_counter = Counter()
    for item in data:
        hop = display_question_graph(
            qid=item["qid"],
            question=item["question"],
            graph_query=item["graph_query"],
            answers=item["answer"]
        )
        hop_counter[hop] += 1

    print("\n📊 Hop Count Summary:")
    for hop in sorted(hop_counter):
        print(f"🔹 {hop}-hop: {hop_counter[hop]} questions")

if __name__ == "__main__":
    process_grailqa(GRAILQA_FILE)