import json

INPUT_FILE = "grailqa.json"
OUTPUT_FILE = "grailqa_with_middle_nodes.json"

def has_middle_node(graph_query):
    for node in graph_query.get("nodes", []):
        node_id = node.get("id", "")
        is_class_node = not node_id.startswith("m.")
        is_not_question_node = node.get("question_node", 0) == 0
        if is_class_node and is_not_question_node:
            return True
    return False

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    middle_node_questions = [q for q in data if has_middle_node(q.get("graph_query", {}))]

    print(f"✅ Total questions: {len(data)}")
    print(f"🔎 Questions with middle nodes: {len(middle_node_questions)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        json.dump(middle_node_questions, fout, indent=2, ensure_ascii=False)
        print(f"💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
