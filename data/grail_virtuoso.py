import json
import re

INPUT_FILE = "grailqa.json"
OUTPUT_FILE = "grailqa_virtuoso.json"

def convert_sparql_to_virtuoso(original_sparql):
    """
    将嵌套 SPARQL 改写为扁平、Virtuoso 可执行格式
    """
    # 提取 body（去掉 prefix 和 select）
    lines = original_sparql.strip().splitlines()
    body_lines = []
    in_body = False
    for line in lines:
        if "WHERE" in line:
            in_body = True
            continue
        if in_body:
            body_lines.append(line.strip())

    body = " ".join(body_lines).strip()

    # 去掉多余括号
    body = body.replace("{", "").replace("}", "")
    body = re.sub(r"\s+", " ", body)

    # 替换 : → ns: (Virtuoso 用 ns: 表示 Freebase)
    body = re.sub(r'\s:([a-zA-Z0-9_.]+)', r' ns:\1', body)
    body = re.sub(r'\(:([a-zA-Z0-9_.]+)', r'(ns:\1', body)

    # 处理 VALUES 行（Virtuoso 不支持 VALUES，需要转为直接引用）
    values_matches = re.findall(r'VALUES\s+\?(\w+)\s*{\s*ns:(m\.[a-zA-Z0-9_]+)\s*}', body)
    for var, mid in values_matches:
        body = re.sub(rf'VALUES\s+\?{var}\s*{{\s*ns:{mid}\s*}}', f'BIND(ns:{mid} AS ?{var})', body)

    # 构建新的 SPARQL
    rewritten = f"""PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x0\nFROM <http://freebase.com>\nWHERE {{ {body} }}"""
    return rewritten

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for entry in data:
        original_sparql = entry.get("sparql_query", "")
        try:
            new_sparql = convert_sparql_to_virtuoso(original_sparql)
            entry["sparql_query_virtuoso"] = new_sparql
        except Exception as e:
            print(f"⚠️ Error converting SPARQL for QID {entry.get('qid')}: {e}")
            entry["sparql_query_virtuoso"] = ""
        new_data.append(entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        json.dump(new_data, fout, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(new_data)} entries and saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
