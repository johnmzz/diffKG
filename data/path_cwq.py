import json
import re
from collections import defaultdict, Counter
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"
file_path = "cwq.json"  # ← 替换为你的 CWQ 数据集路径

def collect_examples_by_hop(file_path, examples_per_hop=3):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    hop_examples = defaultdict(list)
    hop_counts = []

    for item in data:
        question = item.get("question", "N/A")
        sparql = item.get("sparql", "")
        lines = sparql.split('\n')
        triple_lines = [
            line.strip() for line in lines
            if 'ns:' in line and not line.strip().startswith(('PREFIX', 'FILTER', '#'))
        ]
        hop_count = len(triple_lines)
        hop_counts.append(hop_count)
        if len(hop_examples[hop_count]) < examples_per_hop:
            hop_examples[hop_count].append({
                "question": question,
                "triples": triple_lines,
                "sparql": sparql
            })

    return hop_examples, hop_counts

def extract_mids(triple_lines):
    mid_pattern = re.compile(r'ns:(m\.[a-zA-Z0-9_]+)')
    mids = set()
    for line in triple_lines:
        for match in mid_pattern.findall(line):
            mids.add(match)
    return list(mids)

def resolve_entity_labels(mids):
    if not mids:
        return {}
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    mid_values = " ".join([f"ns:{mid}" for mid in mids])
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?mid ?name
    FROM <http://freebase.com>
    WHERE {{
        VALUES ?mid {{ {mid_values} }}
        ?mid ns:type.object.name ?name .
        FILTER(lang(?name) = "en")
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"⚠️ Label query failed: {e}")
        return {}

    label_map = {}
    for b in results["results"]["bindings"]:
        mid = b["mid"]["value"].replace("http://rdf.freebase.com/ns/", "")
        name = b["name"]["value"]
        label_map[mid] = name
    return label_map

def resolve_entity_types(mids):
    if not mids:
        return {}

    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    mid_values = " ".join([f"ns:{mid}" for mid in mids])
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?mid ?type
    FROM <http://freebase.com>
    WHERE {{
        VALUES ?mid {{ {mid_values} }}
        ?mid ns:type.object.type ?type .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"⚠️ Type query failed: {e}")
        return {}

    type_map = defaultdict(set)
    for b in results["results"]["bindings"]:
        mid = b["mid"]["value"].replace("http://rdf.freebase.com/ns/", "")
        t = b["type"]["value"].replace("http://rdf.freebase.com/ns/", "")
        type_map[mid].add(t)
    return {k: list(v) for k, v in type_map.items()}

def annotate_triples(triple_lines, label_map):
    annotated = []
    for line in triple_lines:
        def repl(match):
            mid = match.group(1)
            label = label_map.get(mid, "UNKNOWN")
            return f"ns:{mid} ({label})"
        annotated.append(re.sub(r'ns:(m\.[a-zA-Z0-9_]+)', repl, line))
    return annotated

def get_all_variables(sparql_body):
    vars = set(re.findall(r'\?([a-zA-Z_][a-zA-Z0-9_]*)', sparql_body))
    return sorted(vars)

def force_select_all_vars(original_sparql):
    lines = original_sparql.strip().split('\n')
    prefix_lines = [line for line in lines if line.startswith('PREFIX')]
    body_lines = [line for line in lines if not line.startswith('PREFIX')]
    full_body = "\n".join(body_lines)

    vars = get_all_variables(full_body)
    select_line = "SELECT DISTINCT " + " ".join("?" + v for v in vars)

    body_start = next((i for i, line in enumerate(body_lines) if line.strip().startswith('SELECT')), -1)
    if body_start != -1:
        body_lines[body_start] = select_line
    else:
        body_lines.insert(0, select_line)

    return "\n".join(prefix_lines + body_lines)

def query_variable_bindings_all_vars(sparql_text):
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
    except Exception as e:
        print(f"⚠️ SPARQL execution failed: {e}")

    return {var: list(mids) for var, mids in result_map.items()}

if __name__ == "__main__":
    examples, hop_counts = collect_examples_by_hop(file_path, examples_per_hop=5)

    print("\n📊 所有 SPARQL 跳数分布:")
    hop_summary = Counter(hop_counts)
    for hop in sorted(hop_summary):
        print(f"🔹 {hop}-hop: {hop_summary[hop]} 条")

    print("\n📌 每个 hop 示例及变量绑定:\n")
    for hop in sorted(examples):
        print(f"\n🔸 {hop}-hop 示例:")
        for ex in examples[hop]:
            question = ex["question"]
            triples = ex["triples"]
            original_sparql = ex["sparql"]
            print(f"  🟢 Question: {question}")

            mids = extract_mids(triples)
            triple_label_map = resolve_entity_labels(mids)
            for line in annotate_triples(triples, triple_label_map):
                print(f"     ➤ {line}")

            rewritten_sparql = force_select_all_vars(original_sparql)
            bindings = query_variable_bindings_all_vars(rewritten_sparql)
            all_mids = [mid for mids in bindings.values() for mid in mids]

            label_map = resolve_entity_labels(all_mids)
            type_map = resolve_entity_types(all_mids)

            for var in sorted(bindings):
                for mid in bindings[var][:5]:
                    label = label_map.get(mid)
                    types = type_map.get(mid, [])
                    if label:
                        print(f"        └ ?{var} = {mid} ({label})")
                    else:
                        type_str = f", type = {types[0]}" if types else ""
                        print(f"        └ ?{var} = {mid} (UNKNOWN{type_str})")
