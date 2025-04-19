import json
import re
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"

def extract_mids_and_triples(sparql):
    triple_lines = [
        line.strip() for line in sparql.split('\n')
        if 'ns:' in line and not line.strip().startswith(('PREFIX', 'FILTER', '#'))
    ]
    mid_pattern = re.compile(r'ns:(m\.[a-zA-Z0-9_]+)')
    mids = set(mid_pattern.findall(sparql))
    return triple_lines, list(mids)

def resolve_entity_labels(mids):
    if not mids: return {}
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    values_clause = " ".join(f"ns:{mid}" for mid in mids)
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
        return {
            r['mid']['value'].replace("http://rdf.freebase.com/ns/", ""): r['name']['value']
            for r in results['results']['bindings']
        }
    except:
        return {}

def resolve_entity_types(mids):
    if not mids: return {}
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    values_clause = " ".join(f"ns:{mid}" for mid in mids)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?mid ?type
    FROM <http://freebase.com>
    WHERE {{
      VALUES ?mid {{ {values_clause} }}
      ?mid ns:type.object.type ?type
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = defaultdict(set)
    try:
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            mid = r["mid"]["value"].replace("http://rdf.freebase.com/ns/", "")
            typ = r["type"]["value"].replace("http://rdf.freebase.com/ns/", "")
            result[mid].add(typ)
    except:
        pass
    return {k: list(v) for k, v in result.items()}

def force_select_all_vars(sparql):
    body_vars = sorted(set(re.findall(r"\?([a-z_]\w*)", sparql)))
    if not body_vars: return sparql
    select_line = "SELECT DISTINCT " + " ".join("?" + v for v in body_vars)
    lines = sparql.strip().split('\n')
    prefix = [l for l in lines if l.startswith("PREFIX")]
    body = [l for l in lines if not l.startswith("PREFIX") and not l.startswith("SELECT")]
    return "\n".join(prefix + [select_line] + body)

def query_variable_mids(sparql_text):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(sparql_text)
    sparql.setReturnFormat(JSON)
    result_map = defaultdict(set)
    try:
        results = sparql.query().convert()
        for row in results["results"]["bindings"]:
            for var, val in row.items():
                if val["type"] == "uri" and val["value"].startswith("http://rdf.freebase.com/ns/"):
                    mid = val["value"].replace("http://rdf.freebase.com/ns/", "")
                    result_map[var].add(mid)
    except:
        pass
    return {k: list(v) for k, v in result_map.items()}

# === Main ===
with open("cwq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

updated_data = []

for item in data:
    question = item.get("question", "")
    sparql = item.get("sparql", "")
    triples, mids = extract_mids_and_triples(sparql)
    label_map = resolve_entity_labels(mids)

    print(f"\n🟢 Question: {question}")
    for t in triples:
        annotated = re.sub(r'ns:(m\.[a-zA-Z0-9_]+)', lambda m: f'ns:{m.group(1)} ({label_map.get(m.group(1), "UNKNOWN")})', t)
        print(f"   ➤ {annotated}")

    rewritten = force_select_all_vars(sparql)
    bindings = query_variable_mids(rewritten)
    all_mids = set(mid for mids in bindings.values() for mid in mids)
    label_all = resolve_entity_labels(all_mids)
    type_all = resolve_entity_types(all_mids)

    for var in sorted(bindings):
        for mid in bindings[var]:
            label = label_all.get(mid)
            types = type_all.get(mid, [])
            if label:
                print(f"     └ ?{var} = {mid} ({label})")
            else:
                t_str = f", type = {types[0]}" if types else ""
                print(f"     └ ?{var} = {mid} (UNKNOWN{t_str})")

    # ✅ 直接使用 ?x 的所有绑定值作为答案（覆盖原始）
    final_answer = {
        mid: label_all.get(mid, "UNKNOWN")
        for mid in bindings.get("x", [])
    }

    item["answer"] = final_answer
    updated_data.append(item)

# === 输出新文件 ===
with open("cwq_new.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)

print("\n✅ 已保存更新后的文件为 cwq_new.json")
