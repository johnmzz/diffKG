import json
import re

INPUT_FILE = "grailqa_virtuoso.json"          # 原始包含错误 SPARQL 的文件
OUTPUT_FILE = "grailqa_virtuoso_fixed.json"   # 修复后的输出文件

def fix_values_clause(sparql_query: str) -> str:
    """
    修复 SPARQL 中错误的 VALUES 语法：
    例如：VALUES ?x1 ns:m.xxx → VALUES ?x1 { ns:m.xxx }
    """
    pattern = re.compile(r'VALUES\s+(\?\w+)\s+(ns:m\.[a-zA-Z0-9_]+)\b')
    fixed_query = pattern.sub(r'VALUES \1 { \2 }', sparql_query)
    return fixed_query

def fix_all_sparql_queries(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if "sparql_query" in item:
            item["sparql_query"] = fix_values_clause(item["sparql_query"])

    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)

    print(f"✅ 所有 SPARQL 查询已修复并保存至: {output_path}")

if __name__ == "__main__":
    fix_all_sparql_queries(INPUT_FILE, OUTPUT_FILE)
