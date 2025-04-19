import json

# 文件路径
ORIGINAL_QA_FILE = "SimpleQA.json"
MATCH_FILE = "simpleqa_answer_match.json"
OUTPUT_FILE = "SimpleQA_new.json"

# 加载原始 SimpleQA 数据
with open(ORIGINAL_QA_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# 加载已匹配数据
with open(MATCH_FILE, "r", encoding="utf-8") as f:
    match_data = json.load(f)

# 构建 question 到匹配结果的映射
dict_matches = {}
for entry in match_data:
    if entry.get("status") == "match":
        q = entry["question"].strip().lower()
        for match in entry.get("matched_answers", []):
            dict_matches[q] = {
                "mid": match["answer_mid"],
                "label": match["matched_label"],
                "relation": match["matched_relation"]
            }
            break  # 只取第一个匹配项

# 更新原始数据
updated_data = []
matched_count = 0
unmatched_count = 0

for qa in qa_data:
    question = qa.get("question", "").strip().lower()
    match = dict_matches.get(question)
    if match:
        qa["real_answer"] = {match["mid"]: match["label"]}
        qa["relation"] = match["relation"]
        matched_count += 1
    else:
        qa["real_answer"] = {}
        qa["relation"] = ""
        unmatched_count += 1
    updated_data.append(qa)

# 写入新文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=4, ensure_ascii=False)

# 输出统计信息
print(f"✅ Updated dataset saved to: {OUTPUT_FILE}")
print(f"✅ Matched QAs: {matched_count}")
print(f"❌ Unmatched QAs: {unmatched_count}")
