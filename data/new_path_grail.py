import json
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
GRAILQA_FILE = "grailqa_virtuoso_fixed.json"

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

def execute_sparql(sparql_txt):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print("Query failed:", e)
        return []

def parse_graph_query(graph_query):
    nid_to_node = {}
    triples = []
    all_mids = set()
    middle_nids = []

    for node in graph_query['nodes']:
        nid = node['nid']
        node_id = node['id']
        name = node.get('friendly_name', node_id)
        nid_to_node[nid] = (node_id, name, node['question_node'])
        if node_id.startswith('m.'):
            all_mids.add(node_id)
        elif node['question_node'] == 0:
            middle_nids.append(nid)

    for edge in graph_query['edges']:
        src = edge['start']
        dst = edge['end']
        rel = edge['relation']
        rel_name = edge.get('friendly_name', rel)
        h_id, h_name, _ = nid_to_node[src]
        t_id, t_name, _ = nid_to_node[dst]
        if t_id.startswith('m.'):
            all_mids.add(t_id)
        triples.append((h_id, rel, t_id, h_name, rel_name, t_name))

    return triples, all_mids, middle_nids, nid_to_node

def display_question_with_middle_entities(item):
    qid = item["qid"]
    question = item["question"]
    graph_query = item["graph_query"]
    answers = item["answer"]
    sparql_txt = item.get("sparql_query_virtuoso", "")
    bindings = execute_sparql(sparql_txt)

    triples, all_mids, middle_nids, nid_to_node = parse_graph_query(graph_query)
    mid_var_map = defaultdict(set)

    for row in bindings:
        for var, val in row.items():
            if val["type"] == "uri" and val["value"].startswith("http://rdf.freebase.com/ns/"):
                mid = val["value"].replace("http://rdf.freebase.com/ns/", "")
                mid_var_map[var].add(mid)
                all_mids.add(mid)

    label_map = get_label_for_mids(all_mids)

    print(f"\n🟢 QID: {qid}")
    print(f"🟢 Question: {question}")

    for h_id, rel, t_id, h_name, rel_name, t_name in triples:
        h_label = label_map.get(h_id, h_name) if h_id.startswith('m.') else h_name
        t_label = label_map.get(t_id, t_name) if t_id.startswith('m.') else t_name
        h_disp = f"{h_id} ({h_label})" if h_id.startswith('m.') else h_label
        t_disp = f"{t_id} ({t_label})" if t_id.startswith('m.') else t_name
        print(f"   ➤ {h_disp} --{rel}--> {t_disp}")

    for ans in answers:
        if ans["answer_type"] == "Entity":
            mid = ans["answer_argument"]
            label = ans["entity_name"]
            print(f"        └ Answer: {mid} ({label})")

    for nid, (nid_id, name, qn) in nid_to_node.items():
        if nid_id.startswith("m.") and qn == 0:
            label = label_map.get(nid_id, name)
            print(f"        └ Topic: {nid_id} ({label})")

    for nid in middle_nids:
        class_type, class_name, _ = nid_to_node[nid]
        var = f"x{nid}"
        mids = mid_var_map.get(var, [])
        if not mids:
            continue
        print(f"    🔎 Middle entity for type `{class_type}` (node {nid}):")
        for m in mids:
            label = label_map.get(m, "UNKNOWN")
            print(f"        └ {m} ({label})")

def main():
    with open(GRAILQA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data[:10]:  # remove [:10] for full run
        display_question_with_middle_entities(item)

if __name__ == "__main__":
    main()
