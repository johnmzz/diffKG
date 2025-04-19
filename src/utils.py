import os
import torch
import difflib
from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import pickle
import json
from texttable import Texttable
from functools import lru_cache
import re
from collections import deque, defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON, XML
from openai import OpenAI
import time
import sqlite3
from urllib.error import HTTPError
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


SPARQLPATH = "http://localhost:9890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md


sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


sparql_tail_entities_and_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation ?tailEntity
WHERE {
    ns:%s ?relation ?tailEntity .
}
"""

sparql_head_entities_and_relations = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation ?headEntity
WHERE {
    ?headEntity ?relation ns:%s .
}
"""



def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    t.add_rows(rows)
    print(t.draw())



def initialize_large_database(db_path):
    # connect to the SQLite database file specified by db_path
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()  # create cursor, which allows SQL commands to be withini the DB

        # create a table named subgraphs
        #   question_id: stores the ID of the question being processed
        #   chunk_index: chunk number for large subgraph data is split into multiple pieces
        #   data: stores the actual subgraph information in binary format
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subgraphs (
            question_id TEXT,
            chunk_index INTEGER,
            data BLOB,
            PRIMARY KEY (question_id, chunk_index)
        )
        ''')
        conn.commit()   # save the changes to the SQL DB



def retry_operation(func):
    """Decorator to retry database operations."""
    def wrapper(*args, **kwargs):
        attempts = 0
        max_retries = kwargs.pop('max_retries', 300)
        wait_time = kwargs.pop('wait_time', 6)
        while attempts < max_retries:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    print(f"Database is locked, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    attempts += 1
                else:
                    print("An error occurred:", e)
                    break
        print("Failed after several attempts.")
        return None
    return wrapper

@retry_operation
def delete_data_by_question_id(db_path, question_id):
    with sqlite3.connect(db_path) as conn:
        start = time.time()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM subgraphs WHERE question_id = ?', (question_id,))
        conn.commit()
        # print(f"Data associated with question_id {question_id} has been deleted.")
        # print(f"Time taken to delete data: {time.time() - start} seconds.")

@retry_operation
def save_to_large_db(db_path, question_id, data_dict, chunk_size=256 * 1024 * 1024):
    with sqlite3.connect(db_path) as conn:
        start = time.time()
        cursor = conn.cursor()
        data_blob = pickle.dumps(data_dict)
        for i in range(0, len(data_blob), chunk_size):
            chunk = data_blob[i:i+chunk_size]
            cursor.execute('INSERT INTO subgraphs (question_id, data, chunk_index) VALUES (?, ?, ?)',
                           (question_id, chunk, i // chunk_size))
            conn.commit()
        # print("Data saved to database.")
        # print(f"Time taken to save data: {time.time() - start} seconds.")

# Retrieves and reconstructs stored data from an SQLite database based on a question_id
@retry_operation
def load_from_large_db(db_path, question_id):
    with sqlite3.connect(db_path) as conn:  # connect with DB
        start = time.time()
        cursor = conn.cursor()

        # Retrieves the "data" column of records where "question_id" matches
        cursor.execute('SELECT data FROM subgraphs WHERE question_id = ? ORDER BY chunk_index', (question_id,))
        
        # Concatenates all retrieved data values into a single binary blob
        data_blob = b''.join(row[0] for row in cursor if row[0] is not None)
        
        if not data_blob:
            # print("No data found or data is empty.")
            return None
        
        # Deserializes (unpickles) the binary blob back into its original Python object.
        try:
            result = pickle.loads(data_blob)
            return result
        except EOFError as e:
            print(f"EOFError when unpickling data: {e}")
            return None


def query_label_from_kg(mid, sparql_endpoint="http://localhost:9890/sparql"):
    sparql = SPARQLWrapper(sparql_endpoint)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?name
    FROM <http://freebase.com>
    WHERE {{
        ns:{mid} ns:type.object.name ?name .
        FILTER (lang(?name) = "en")
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        result = sparql.query().convert()
        bindings = result.get("results", {}).get("bindings", [])
        return bindings[0]["name"]["value"] if bindings else None
    except:
        return None



def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        ID = 'ID'

    elif dataset_name == 'cwq_multi':
        with open('../data/cwq_multi.json',encoding='utf-8') as f:
            datas = json.load(f)
        ID = 'ID'
        question_string = 'question'

    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        ID = "QuestionId"
        # answer = ""

    elif dataset_name == 'webqsp_multi':
        with open('../data/webqsp_multi.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'

    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
        ID = 'qid'
        
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
        ID = 'question'
        answer_string = 'answer'

    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        ID = 'question'
        question_string = 'question'

    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string, ID



def check_answerlist(dataset_name, origin_data, topic_entity):
    final_answer_list = []
    answer_list = []
    # origin_data = [j for j in ground_truth_datas if j[question_string] == ori_question]
    if dataset_name == 'cwq':
        for k, v in origin_data["answer"].items():
            answer_list.append(
                {"mid": k, "label": v}
            )

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for parse in answers:
            for ans in parse['Answers']:
                if ans.get("AnswerType") == "Entity":
                    mid = ans.get("AnswerArgument")
                    label = query_label_from_kg(mid)
                    if label:
                        answer_list.append(
                            {"mid": mid, "label": label}
                        )

    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if answer.get("answer_type") == "Entity":
                mid = answer["answer_argument"]
                label = query_label_from_kg(mid)
                if label:
                    answer_list.append(
                        {"mid": mid, "label": label}
                    )

    elif dataset_name == 'simpleqa':
        for k,v in origin_data["real_answer"].items():
            answer_list.append(
                {"mid": k, "label": v, "relation": origin_data["relation"]}
            )

    elif dataset_name == 'webquestions':
        for k,v in origin_data["real_answers"].items():
            answer_list.append(
                {"mid": k, "label": v["label"], "relation": v["relation"]}
            )

    return [dict(t) for t in {tuple(sorted(d.items())) for d in answer_list}]



def initialize_large_database(db_path):
    # connect to the SQLite database file specified by db_path
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()  # create cursor, which allows SQL commands to be withini the DB

        # create a table named subgraphs
        #   question_id: stores the ID of the question being processed
        #   chunk_index: chunk number for large subgraph data is split into multiple pieces
        #   data: stores the actual subgraph information in binary format
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS subgraphs (
            question_id TEXT,
            chunk_index INTEGER,
            data BLOB,
            PRIMARY KEY (question_id, chunk_index)
        )
        ''')
        conn.commit()   # save the changes to the SQL DB



# model_id: Huggingface 模型的字符串路径，例如 "openchat/openchat-3.5-0106"
# cache_dir: 本地模型缓存目录，将模型下载并保存在这里
def init_LLM(model_id: str, cache_dir: str = "/data1/zhuom/LLM"):
    """
    加载指定模型，自动适配 prompt 并执行测试。
    支持多个 Huggingface 本地模型。
    """
    print(f"📦 准备加载模型: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    print("✅ 模型加载完成")

    # ========= Prompt 构造器 =========
    def format_prompt(user_input: str) -> str:
        # ChatML 样式（OpenChat, Mistral, Nous, Mixtral, Yi, Qwen 等）
        if any(key in model_id.lower() for key in [
            "openchat", "mistral", "mixtral", "nous", "zephyr", "starling", "yi", "qwen", "gemma"
        ]):
            return (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        # LLaMA 2/3 系列 INST 格式
        elif "llama" in model_id.lower():
            return f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{user_input} [/INST]"
        else:
            raise NotImplementedError(f"❌ 未支持该模型的 prompt 格式: {model_id}")

    # ========= 自动选择测试问题 =========
    test_question = {
        "openchat/openchat-3.5-0106": "What is the capital of France?",
        "mistralai/Mistral-7B-Instruct-v0.3": "What is the capital of China?",
        "NousResearch/Nous-Hermes-2-Mistral-7B-DPO": "What is the capital of Japan?",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "What are the main causes of climate change?",
        "meta-llama/Llama-2-13b-chat-hf": "Who wrote the novel '1984'?",
        "meta-llama/Meta-Llama-3-8B-Instruct": "Explain how a neural network learns.",
        "google/gemma-7b-it": "What is the boiling point of water?",
        "HuggingFaceH4/zephyr-7b-beta": "How does photosynthesis work?",
        "berkeley-nest/Starling-LM-7B-alpha": "What is the tallest mountain on Earth?",
        "Qwen/Qwen1.5-7B-Chat": "How many continents are there?",
        "01-ai/Yi-1.5-9B-Chat": "What is the Pythagorean theorem?"
    }.get(model_id, "What is the capital of France?")  # 默认 fallback

    prompt = format_prompt(test_question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 设置生成配置（避免 warning）
    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.do_sample = True
    gen_config.temperature = 0.4
    gen_config.top_p = 0.9
    gen_config.max_new_tokens = 128
    gen_config.pad_token_id = tokenizer.eos_token_id

    print(f"🧠 测试问题: {test_question}")
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    # 解码并只保留第一轮 assistant 回答
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    elif "[/INST]" in prompt and "###" in response:
        response = response.split("###")[0]
    print("\n💬 模型回复：")
    print(response.strip())

    return model, tokenizer



def run_LLM(prompt, model_name, model=None, tokenizer=None, temperature=0.4):
    result = ''

    # ========== OpenAI GPT 模型 ==========
    if "gpt" in model_name.lower():             # openai_api_base = "http://localhost:8000/v1"
        openai_api_key = "your_api_keys"
        if model_name == "gpt4":
            model_engine = "gpt-4-turbo"
        else:
            model_engine = "gpt-3.5-turbo-0125"

        client = OpenAI(api_key=openai_api_key)     # create an OpenAI client for sending request

        # 构造 OpenAI 的 Chat 请求格式（messages = [system msg + user prompt]）
        system_message = "You are an AI assistant that helps people find information."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # 用接口发起对话，最多尝试 3 次，每次失败间隔 2 秒。
        try_time = 0
        while try_time < 3:
            try:
                response = client.chat.completions.create(
                    model=model_engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=512,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                result = response.choices[0].message.content
                break
            except Exception as e:
                print(f"OpenAI error: {e}")
                print("Retrying in 2 seconds...")
                try_time += 1
                time.sleep(2)
                
    # ========== 本地模型 ==========
    else:
        if model is None or tokenizer is None:
            raise ValueError("本地模型调用时必须传入 model 和 tokenizer")

        # === Prompt 包装：根据模型类型选择格式 ===
        model_name_lower = model_name.lower()

        if any(key in model_name_lower for key in [
            "openchat", "mistral", "mixtral", "nous", "zephyr", "starling", "yi", "qwen", "gemma"
        ]):
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif "llama" in model_name_lower:
            # LLaMA 2/3 格式
            prompt = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            raise NotImplementedError(f"❌ 未支持的本地模型 prompt 格式: {model_name}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate config
        gen_config = GenerationConfig.from_model_config(model.config)
        gen_config.do_sample = True
        gen_config.temperature = temperature
        gen_config.top_p = 0.9
        gen_config.max_new_tokens = 512
        gen_config.pad_token_id = tokenizer.eos_token_id

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)

        # Decode output
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # 截断模型回复，只保留第一轮
        if "<|im_end|>" in decoded:
            result = decoded.split("<|im_end|>")[0].strip()
        elif "[/INST]" in prompt and "###" in decoded:
            result = decoded.split("###")[0].strip()
        else:
            result = decoded.strip()

    print(f"\n{model_name} response:\n{result}\n")
    return result



# 从 LLM 返回的文本（包含推理链 CoT）中提取:
# - 推理路径长度（path_length）：用于预测图探索起始深度 D_predict
# - 推理链文本（thinking_cot_line）：用于指导路径排序、实体排序等。
def extract_path_length_from_text(text):
    back_up = text

    # 基于 " - " 拆分关系链（实体-关系-实体...）
    tokens = re.split(r'\s*-\s*', text.strip())

    # 计算路径长度
    # 每条路径结构是：Entity - Relation - Entity，所以 每跳需要两个 token（一个关系和一个目标实体）
    path_length = (len(tokens) - 1) // 2

    # e抽取推理链 CoT 行
    match = re.search(r'cot\s*:\s*(.*)', back_up, re.IGNORECASE)

    # 输出结果
    if match:
        thinking_cot_line = match.group(1).strip()
        # print('提取的文本是：')
        # print(thinking_cot_line)
    else:
        print('cannot find the cot line')

    return path_length, thinking_cot_line


# 从 LLM 的返回文本中提取 拆分后的子问题（split questions）
# Return: ["question1", "question2", ...]
def extract_split_questions(text):
    lines = text.strip().split('\n')
    questions = []  # store extracted split questions

    for line in lines:
        line_no_spaces = line.replace(' ', '')

        # 如果该行包含 split，提取冒号后面的子问题
        if re.search(r'split', line_no_spaces, re.IGNORECASE):
            parts = line.split(':', 1)
            if len(parts) > 1:
                question = parts[1].strip()
                questions.append(question)
            else:
                # 如果没有 ':'，整个行作为问题
                questions.append(line.strip())

    return questions


# 根据 LLM 生成的推理链（CoT reasoning line），对 topic entities 进行顺序重排，以匹配 LLM 的思维顺序。
def reorder_entities(cot_line, topic_entity_dict):
    entity_positions = []   # 用于存储 (position, entity_name) 元组。

    for entity in topic_entity_dict:
        score, position = find_best_matching_substring(entity, cot_line)

        # Assign a high position if no match is found
        if position != -1:
            entity_positions.append((position, entity))
        else:
            entity_positions.append((float('inf'), entity)) # assign (inf, entity) if not found

    # Sort entities based on their positions in cot_line
    entity_positions.sort()

    sorted_entities = [entity for position, entity in entity_positions]
    return sorted_entities


# Finds the best matching substring of an entity within a given CoT-line.
# Returning: the best match (highest similarity score and starting position)
def find_best_matching_substring(entity, cot_line):
    len_entity = len(entity)
    len_cot = len(cot_line)

    # Consider substrings within reasonable lengths
    min_len = max(1, len_entity // 2)
    max_len = min(len_cot, len_entity * 2)

    best_score = 0
    best_start = -1

    for length in range(min_len, max_len + 1):      # iterate over all string length
        for start in range(len_cot - length + 1):   # sliding window over 
            substring = cot_line[start:start + length]
            score = difflib.SequenceMatcher(None, entity, substring).ratio()    # compute similarity ratio
            if score > best_score:
                best_score = score
                best_start = start

    return best_score, best_start



# 给定实体 ID（如 "m.02jx3"），通过 SPARQL 查询知识图谱，返回这个实体的名称（例如 "Barack Obama"）
# 使用缓存加速查询，最近使用的最多 1024 个结果会被缓存，避免频繁重复向 KG 发起相同请求
@lru_cache(maxsize=1024)
def id2entity_name_or_type(entity_id):
    init_id = entity_id

    # 构造 SPARQL 查询语句
    entity_id = sparql_id % (format(entity_id), format(entity_id))

    # prepare SPARQL query
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(entity_id)
    sparql.setReturnFormat(JSON)

    # send SPARQL query
    results = []
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            break
            # return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)
            time.sleep(2)  # Sleep for 2 seconds before retrying
            attempts += 1  

    if attempts == 3:
        print("Failed to execute after multiple attempts.")

    # 处理查询结果
    if len(results["results"]["bindings"]) == 0:    # if no results are found, return "Unnamed Entity"
        return "Unnamed Entity"
    else:
        # 优先选择英文名（xml:lang == 'en'）
        english_results = [result['tailEntity']['value'] for result in results["results"]["bindings"] if result['tailEntity'].get('xml:lang') == 'en']
        if english_results:
            return english_results[0]  # Return the first English result

        #   If no English labels are found, checks for names that contain only letters and numbers (ignores symbols)
        alphanumeric_results = [result['tailEntity']['value'] for result in results["results"]["bindings"]
                                if re.match("^[a-zA-Z0-9 ]+$", result['tailEntity']['value'])]
        if alphanumeric_results:
            return alphanumeric_results[0]  # Return the first alphanumeric result
        
        # 如果查询失败或结果不合法，则返回 "Unnamed Entity"。
        return "Unnamed Entity"



# 一组实体出发，并行地向外扩展一跳邻居，更新图结构和实体信息
def explore_graph_from_one_topic_entities(current_entities, graph, entity_names, explored_entities, all_entities, limit_per_entity=100):
    """
    current_entities: set(str), 当前轮次中要扩展的实体集合（Freebase MID）
    graph: dict, 当前正在构建的图结构
    entity_names: dict, MID → 实体名称
    exlored_entities: set(str), 已所有已探索过的实体，防止重复
    all_entities: set(str), 所有发现的实体（全局集合）
    answer_name: list(str), 正确答案的实体名称
    """
    storage_lock = Lock()
    new_entities = set()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(updated_search_relations_and_entities_combined_1, entity, limit_per_entity): entity
            for entity in current_entities
        }

        explored_entities.update(current_entities)

        for future in as_completed(futures):
            results = future.result()
            entity = futures[future]

            for result in results:
                rel = result["relation"]
                neighbor = result["connectedEntity"]
                name = result["connectedEntityName"]
                direction = result["direction"]

                if not neighbor.startswith("m.") or neighbor in explored_entities:
                    continue

                with storage_lock:
                    entity_names[neighbor] = name

                    if entity not in graph:
                        graph[entity] = {}

                    if neighbor not in graph:
                        graph[neighbor] = {}

                    if neighbor not in graph[entity]:
                        graph[entity][neighbor] = {"forward": set(), "backward": set()}
                        
                    if entity not in graph[neighbor]:
                        graph[neighbor][entity] = {"forward": set(), "backward": set()}

                    if direction == "tail":
                        graph[entity][neighbor]["forward"].add(rel)
                        graph[neighbor][entity]["backward"].add(rel)
                    else:
                        graph[entity][neighbor]["backward"].add(rel)
                        graph[neighbor][entity]["forward"].add(rel)

                new_entities.add(neighbor)

    new_entities.difference_update(explored_entities)
    all_entities.update(new_entities)
    current_entities = new_entities

    return graph, all_entities, explored_entities, current_entities, entity_names



# 从一个实体出发，在 Freebase 知识图谱中查询与之连接的所有三元组，并返回清洗后的结构化结果。
def updated_search_relations_and_entities_combined_1(entity_id, limit=100):
    """
    查询实体的邻居，仅保留有 label 的，限制返回数量。
    """
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?relation ?connectedEntity ?connectedEntityName ?direction
    WHERE {{
        {{
            ns:{entity_id} ?relation ?connectedEntity .
            ?connectedEntity ns:type.object.name ?name .
            FILTER(lang(?name) = 'en')
            BIND(?name AS ?connectedEntityName)
            BIND("tail" AS ?direction)
        }}
        UNION
        {{
            ?connectedEntity ?relation ns:{entity_id} .
            ?connectedEntity ns:type.object.name ?name .
            FILTER(lang(?name) = 'en')
            BIND(?name AS ?connectedEntityName)
            BIND("head" AS ?direction)
        }}
    }}
    LIMIT {limit}
    """
    results = execute_sparql(query)
    return replace_prefix1(results)




# 将 http://rdf.freebase.com/ns/ 前缀清洗掉，保留 m.02jx3 这样的简写 MID。
def replace_prefix1(data):
    if data is None:
        print("Warning: No data available to process in replace_prefix1.")
        return []

    # Returns a list of cleaned dictionaries.
    return [{key: value['value'].replace("http://rdf.freebase.com/ns/", "") for key, value in result.items()} for result in data]



def execute_sparql(sparql_txt):
    """
    Sends a SPARQL query to a Freebase/Wikidata endpoint and retrieves structured data in JSON format.
    Returns a list of dictionaries (from "bindings") or None if the query fails.
    """
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery("define sql:big-data-const 0\n" + sparql_txt)  # ✅ 不再添加 LIMIT 3000
    sparql.setReturnFormat(JSON)
    
    attempts = 0
    while attempts < 3:
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)
            time.sleep(2)
            attempts += 1

    print("Failed to execute after multiple attempts.")
    return None



# 在最多 max_depth 跳内，从多个 topic entities 同时出发，搜索邻接实体，并构建一个包含这些实体、关系、方向的图结构，
# 直到找到所有答案实体并判断这些 topic entities 是否互相连通。
# entity_ids: 多个 topic entity 的 Freebase ID（例如 "m.02jx3"）
# answer_name: 正确答案的名字列表（用于判断是否找到答案）
def explore_graph_from_entities_by_hop_neighbor_1(entity_ids, max_depth=5, answer_name=[]):
    current_entities = set(entity_ids)  # 当前这一层的探索节点
    all_entities = set(entity_ids)      # 总共见过的节点
    found_answer = False                # 是否找到所有答案

    entity_names = {entity: id2entity_name_or_type(entity) for entity in entity_ids}  # 默认所有初始实体名称为"unnamedentity"
    graph = {entity: {} for entity in all_entities}  # 初始化图

    storage_lock = Lock()               # 创建线程安全锁
    
    answer_name_set = set(answer_name) 
    empty_set = set()                   # 用于判断已找到的答案实体
    if len(entity_ids) == 1:
        connect = True
    else:
        connect = False
    hopnumber = 5

    #  多轮 BFS：从多个实体一起出发，探索邻居直到 max_depth
    for depth in range(1, max_depth + 1):
        print(f"Exploring entities at depth {depth}...")
        start = time.time()
        new_entities = set()

        # 用多线程并发搜索邻居实体
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(search_relations_and_entities_combined_1, entity): entity for entity in current_entities}

            # 对返回的结果处理图结构 + 筛选答案
            for future in as_completed(futures):
                results = future.result()
                entity = futures[future]
                for result in results:
                    relation, connected_entity, connected_name, direction = result['relation'], result['connectedEntity'], result['connectedEntityName'], result['direction']

                    if connected_entity.startswith("m."):
                        # 检查 connected_name 是否在答案集中，如果是，则标记为答案实体
                        if connected_name in answer_name_set:
                            empty_set.add(connected_entity)
                            if len(empty_set) == len(answer_name_set):
                                found_answer = True
                        # 更新图
                        with storage_lock:
                            # 更新或添加实体名称
                            entity_names[connected_entity] = connected_name
                            # 确保图中包含相关实体和关系
                            if entity not in graph:
                                graph[entity] = {}
                            if connected_entity not in graph:
                                graph[connected_entity] = {}
                            if connected_entity not in graph[entity]:
                                graph[entity][connected_entity] = {'forward': set(), 'backward': set()}
                            if entity not in graph[connected_entity]:
                                graph[connected_entity][entity] = {'forward': set(), 'backward': set()}

                            # 更新关系
                            if direction == "tail":
                                graph[entity][connected_entity]['forward'].add(relation)
                                graph[connected_entity][entity]['backward'].add(relation)
                            else:  # direction is "head"
                                graph[entity][connected_entity]['backward'].add(relation)
                                graph[connected_entity][entity]['forward'].add(relation)
                        new_entities.add(connected_entity)


        new_entities.difference_update(all_entities)
        all_entities.update(new_entities)
        current_entities = new_entities
        end = time.time()
        print(f"Time taken to explore depth {depth}: {end - start:.2f} seconds")
        if connect == False:
            connect = are_entities_connected(graph, entity_ids, all_entities)
            if connect:
                print(f"All entities are connected within {depth} hops.")
                hopnumber = depth

        if found_answer and connect:
            return (found_answer, graph, hopnumber, all_entities, current_entities, entity_names, connect)

    print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (found_answer, graph, hopnumber, all_entities, current_entities, entity_names, connect)



# 所有 topic entities 是否处于同一个连通子图中
def are_entities_connected(graph, total_entities, all_entities):
    """
    Check if starting from the first entity in total_entities, all other entities in total_entities can be visited.
    graph: Dictionary with entity as key and another dictionary {connected_entity: {'forward': set(), 'backward': set()}} as value.
    total_entities: Set of initial entities to check connectivity from.
    """
    # 如果没有实体，则连通
    if not total_entities:
        return True

    total_entities_set = set(total_entities)

    # 如果只有一个实体，则联通
    if len(total_entities_set) == 1:
        return True

    # 从任意一个实体出发做 BFS
    start_entity = next(iter(total_entities_set))
    visited = set()
    queue = deque([start_entity])

    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            # Early termination check
            if total_entities_set.issubset(visited):
                return True

            # Add connected entities to the queue
            for connected_entity, relations in graph[current].items():
                if connected_entity not in visited:
                    queue.append(connected_entity)

    # Final check in case not all entities are connected
    return False



# def retrieve_answer_from_kg_simpleqa(topic_mid, relation_keyword, expected_answer_text):
#     sparql = SPARQLWrapper(SPARQL_ENDPOINT)

#     # Step 1: 获取 topic_mid 所有谓词
#     query1 = f"""
#     PREFIX ns: <http://rdf.freebase.com/ns/>
#     SELECT DISTINCT ?p
#     FROM <http://freebase.com>
#     WHERE {{
#         ns:{topic_mid} ?p ?o .
#     }}
#     LIMIT 100
#     """
#     sparql.setQuery(query1)
#     sparql.setReturnFormat(JSON)
#     try:
#         results = sparql.query().convert()
#         predicates = [
#             b["p"]["value"].replace("http://rdf.freebase.com/ns/", "")
#             for b in results["results"]["bindings"]
#         ]
#     except Exception:
#         return None, None, None, [], {}

#     # Step 2: 模糊匹配 relation keyword
#     matched_preds = [p for p in predicates if relation_keyword.lower() in p.lower().split(".")[-1]]
#     tried_answers = {}

#     if not matched_preds:
#         return None, None, None, matched_preds, tried_answers

#     # Step 3: 遍历谓词，找出符合 expected_answer_text 的 entity
#     for predicate in matched_preds:
#         query2 = f"""
#         PREFIX ns: <http://rdf.freebase.com/ns/>
#         SELECT DISTINCT ?mid ?name
#         FROM <http://freebase.com>
#         WHERE {{
#             ns:{topic_mid} ns:{predicate} ?mid .
#             OPTIONAL {{ ?mid ns:type.object.name ?name . FILTER (lang(?name) = "en") }}
#         }}
#         LIMIT 200
#         """
#         sparql.setQuery(query2)
#         sparql.setReturnFormat(JSON)

#         try:
#             res = sparql.query().convert()
#             answer_list = []
#             for b in res["results"]["bindings"]:
#                 mid = b["mid"]["value"].replace("http://rdf.freebase.com/ns/", "")
#                 label = b["name"]["value"].strip() if "name" in b else None
#                 answer_list.append((mid, label))

#                 if label and label.lower() == expected_answer_text.lower():
#                     return mid, label, predicate, matched_preds, tried_answers

#             tried_answers[predicate] = answer_list

#         except Exception:
#             continue

#     # Step 4: fallback，选择第一个非空 relation 的第一个 entity
#     for pred, items in tried_answers.items():
#         for mid, label in items:
#             if label:  # ✅ 只返回 label 非空的
#                 return mid, label, pred, matched_preds, tried_answers

#     return None, None, None, matched_preds, tried_answers



# def is_compound_node(mid):
#     # 判断某个mid是否是复合类型节点（没有name但有type）
#     sparql = SPARQLWrapper(SPARQLPATH)
#     query = f"""
#     PREFIX ns: <http://rdf.freebase.com/ns/>
#     ASK {{
#       ns:{mid} ns:type.object.name ?name .
#     }}
#     """
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)
#     try:
#         has_name = sparql.query().convert()["boolean"]
#         return not has_name  # 没有name → 可能是compound
#     except Exception:
#         return False



# def search_entity_neighbors(mid, relation, include_compound=True):
#     sparql = SPARQLWrapper(SPARQLPATH)

#     # Forward + backward 1-hop neighbors
#     query = f"""
#     PREFIX ns: <http://rdf.freebase.com/ns/>
#     SELECT DISTINCT ?e
#     FROM <http://freebase.com>
#     WHERE {{
#         {{
#             ns:{mid} ns:{relation} ?e .
#         }}
#         UNION
#         {{
#             ?e ns:{relation} ns:{mid} .
#         }}
#     }}
#     """
#     sparql.setQuery(query)
#     sparql.setReturnFormat(JSON)

#     try:
#         results = sparql.query().convert()
#         neighbors = [
#             b["e"]["value"].replace("http://rdf.freebase.com/ns/", "")
#             for b in results["results"]["bindings"]
#             if b["e"]["value"].startswith("http://rdf.freebase.com/ns/m.")
#         ]
#     except Exception as e:
#         print(f"SPARQL query failed for {mid}, relation {relation}: {e}")
#         return []

#     final_neighbors = set()

#     for n in neighbors:
#         if not include_compound:
#             final_neighbors.add(n)
#             continue

#         if is_compound_node(n):
#             final_neighbors.add(n)  # ✅ 保留 compound 本身

#             # 查询 compound 节点的两侧邻居（不止 forward，还包括 backward）
#             query2 = f"""
#             PREFIX ns: <http://rdf.freebase.com/ns/>
#             SELECT DISTINCT ?e2
#             FROM <http://freebase.com>
#             WHERE {{
#                 {{
#                     ns:{n} ?r ?e2 .
#                     FILTER (STRSTARTS(STR(?e2), "http://rdf.freebase.com/ns/m."))
#                     FILTER (?e2 != ns:{mid})
#                 }}
#                 UNION
#                 {{
#                     ?e2 ?r ns:{n} .
#                     FILTER (STRSTARTS(STR(?e2), "http://rdf.freebase.com/ns/m."))
#                     FILTER (?e2 != ns:{mid})
#                 }}
#             }}
#             """
#             sparql.setQuery(query2)
#             sparql.setReturnFormat(JSON)
#             try:
#                 res2 = sparql.query().convert()
#                 for b in res2["results"]["bindings"]:
#                     e2 = b["e2"]["value"].replace("http://rdf.freebase.com/ns/", "")
#                     final_neighbors.add(e2)
#             except Exception as e:
#                 print(f"SPARQL compound expansion failed for {n}: {e}")
#         else:
#             final_neighbors.add(n)

#     return list(final_neighbors)




def search_entity_neighbors(mid, relation):
    sparql = SPARQLWrapper(SPARQLPATH)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?e
    FROM <http://freebase.com>
    WHERE {{
        {{
            ns:{mid} ns:{relation} ?e .
        }}
        UNION
        {{
            ?e ns:{relation} ns:{mid} .
        }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        neighbors = [
            b["e"]["value"].replace("http://rdf.freebase.com/ns/", "")
            for b in results["results"]["bindings"]
            if b["e"]["value"].startswith("http://rdf.freebase.com/ns/m.")
        ]
        return neighbors
    except Exception as e:
        print(f"SPARQL query failed for {mid}, relation {relation}: {e}")
        return []





def get_label_for_mids(mids):
    if isinstance(mids, str):
        mids = [mids]  # 如果传入的是单个mid，自动转换为列表

    sparql = SPARQLWrapper(SPARQLPATH)
    values_clause = " ".join(f"ns:{mid}" for mid in mids)
    
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?mid ?name
    WHERE {{
      VALUES ?mid {{ {values_clause} }}
      ?mid ns:type.object.name ?name .
      FILTER(lang(?name) = "en")
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    labels = {}
    try:
        results = sparql.query().convert()
        for b in results["results"]["bindings"]:
            mid = b["mid"]["value"].replace("http://rdf.freebase.com/ns/", "")
            labels[mid] = b["name"]["value"]
    except Exception as e:
        print("SPARQL query failed:", e)

    return labels



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
    sparql = SPARQLWrapper(SPARQLPATH)
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



def resolve_entity_types(mids):
    if not mids:
        return {}

    sparql = SPARQLWrapper(SPARQLPATH)
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
    sparql = SPARQLWrapper(SPARQLPATH)
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



def construct_subgraph(dataset, data, topic_entity, entity_names, question_real_answer, depth):
    print("--------- Construct Subgraph ---------")
    explored_entities = set()                           # 已经被访问过的实体
    next_explore_entities = set(topic_entity.keys())    # 当前这轮 BFS 将要扩展的实体（初始为 topic entities）
    all_entities = set(topic_entity.keys())             # 记录图中所有访问到的实体
    gt_entities = set()

    # 临时维护的子图（邻接表结构）
    graph = {entity: {} for entity in topic_entity.keys()}

    # ------------------------- SimpleQA -------------------------
    if dataset == "simpleqa":
        for answer in question_real_answer:
            answer_label = answer['label']
            answer_mid = answer['mid']
            answer_relation = answer['relation']

            entity_names[answer_mid] = answer_label

            print(f"✔️ 添加答案: {answer_label} ({answer_mid}) via {answer_relation}")

            # 放入 explored_entities 和 all_entities
            explored_entities.add(answer_mid)
            all_entities.add(answer_mid)

            # 建立图结构中对应的节点
            if answer_mid not in graph:
                graph[answer_mid] = {}

            for topic_id in topic_entity:
                if topic_id not in graph:
                    graph[topic_id] = {}
                if answer_mid not in graph[topic_id]:
                    graph[topic_id][answer_mid] = {'forward': set(), 'backward': set()}
                if topic_id not in graph[answer_mid]:
                    graph[answer_mid][topic_id] = {'forward': set(), 'backward': set()}

                # 添加关系（默认为 topic -> answer 是 forward）
                graph[topic_id][answer_mid]['forward'].add(answer_relation)
                graph[answer_mid][topic_id]['backward'].add(answer_relation)

    # ------------------------- WebQuestions -------------------------
    elif dataset == "webquestions":
        for answer in question_real_answer:
            answer_label = answer['label']
            answer_mid = answer['mid']
            answer_relation = answer['relation']

            entity_names[answer_mid] = answer_label

            print(f"✔️ 添加答案: {answer_label} ({answer_mid}) via {answer_relation}")

            # 放入 explored_entities 和 all_entities
            explored_entities.add(answer_mid)
            all_entities.add(answer_mid)

            # 建立图结构中对应的节点
            if answer_mid not in graph:
                graph[answer_mid] = {}

            for topic_id in topic_entity:
                if topic_id not in graph:
                    graph[topic_id] = {}
                if answer_mid not in graph[topic_id]:
                    graph[topic_id][answer_mid] = {'forward': set(), 'backward': set()}
                if topic_id not in graph[answer_mid]:
                    graph[answer_mid][topic_id] = {'forward': set(), 'backward': set()}

                # 添加关系（默认为 topic -> answer 是 forward）
                graph[topic_id][answer_mid]['forward'].add(answer_relation)
                graph[answer_mid][topic_id]['backward'].add(answer_relation)

    # ------------------------- GrailQA -------------------------
    elif dataset == "grailqa":
        graph_query = data.get("graph_query", {})
        if not graph_query:
            print("⚠️ No graph_query found in this GrailQA item.")
            return graph, entity_names
        
        nid_to_id = {}   # nid → 实体 ID（mid / friendly_name）
        id_to_name = {}  # 实体 ID → 展示名称
        nid_type = {}    # nid → node_type 标记：'topic' / 'answer' / 'intermediate'

        # Step 1: 处理所有节点，构造 nid_to_id / id_to_name
        for node in graph_query.get("nodes", []):
            nid = node["nid"]
            raw_id = node["id"]
            friendly = node.get("friendly_name", raw_id)
            qnode_flag = node.get("question_node", 0)

            if raw_id.startswith("m."):  # topic entity
                nid_to_id[nid] = raw_id
                id_to_name[raw_id] = topic_entity.get(raw_id, friendly)  # topic_entity 中已有 label
                # entity_names[raw_id] = id_to_name[raw_id]
                # all_entities.add(raw_id)
                # explored_entities.add(raw_id)
                nid_type[nid] = "topic"
                if raw_id not in graph:
                    graph[raw_id] = {}

            elif qnode_flag == 1:  # answer node
                nid_type[nid] = "answer"
                # 多个 answer 实体会映射到同一个 nid
                # 先不加实体，等下统一在 edge 处理中展开到多个实体

            else:  # 中间节点
                nid_to_id[nid] = friendly
                id_to_name[friendly] = friendly
                entity_names[friendly] = friendly
                nid_type[nid] = "intermediate"
                if friendly not in graph:
                    graph[friendly] = {}

        # Step 2: 遍历边
        for edge in graph_query.get("edges", []):
            src_nid = edge["start"]
            dst_nid = edge["end"]
            rel = edge["relation"]
            rel_name = edge.get("friendly_name", rel)

            src_type = nid_type.get(src_nid)
            dst_type = nid_type.get(dst_nid)

            # Case 1: answer node 参与（需要展开为多个实体）
            if src_type == "answer" or dst_type == "answer":
                for answer in data["answer"]:
                    ans_mid = answer["answer_argument"]
                    ans_name = answer["entity_name"]
                    entity_names[ans_mid] = ans_name
                    all_entities.add(ans_mid)
                    explored_entities.add(ans_mid)
                    if ans_mid not in graph:
                        graph[ans_mid] = {}

                    # 替换 answer node 为真实实体 ID
                    src_id = ans_mid if src_type == "answer" else nid_to_id[src_nid]
                    dst_id = ans_mid if dst_type == "answer" else nid_to_id[dst_nid]

                    # 名称映射（中间节点可能用 friendly name）
                    h_label = entity_names.get(src_id, src_id)
                    t_label = entity_names.get(dst_id, dst_id)

                    # 初始化图结构
                    if dst_id not in graph[src_id]:
                        graph[src_id][dst_id] = {'forward': set(), 'backward': set()}
                    if src_id not in graph[dst_id]:
                        graph[dst_id][src_id] = {'forward': set(), 'backward': set()}

                    graph[src_id][dst_id]['forward'].add(rel)
                    graph[dst_id][src_id]['backward'].add(rel)

                    print(f"➤ {src_id} ({h_label}) --{rel}--> {dst_id} ({t_label})") 

            else:
                h_id = nid_to_id[src_nid]
                t_id = nid_to_id[dst_nid]
                h_label = entity_names.get(h_id, h_id)
                t_label = entity_names.get(t_id, t_id)

                if h_id not in graph:
                    graph[h_id] = {}
                if t_id not in graph:
                    graph[t_id] = {}
                if t_id not in graph[h_id]:
                    graph[h_id][t_id] = {'forward': set(), 'backward': set()}
                if h_id not in graph[t_id]:
                    graph[t_id][h_id] = {'forward': set(), 'backward': set()}

                graph[h_id][t_id]['forward'].add(rel)
                graph[t_id][h_id]['backward'].add(rel)

                print(f"➤ {h_id} ({h_label}) --{rel}--> {t_id} ({t_label})")

        # Step 3: 替换中间节点的 friendly name 为真实 mid (修正版)
        # 首先，收集中间节点与它的关联边
        intermediate_edges = defaultdict(list)
        for edge in graph_query["edges"]:
            s, t = edge["start"], edge["end"]
            if nid_type.get(s) == "intermediate":
                intermediate_edges[nid_to_id[s]].append(edge)
            if nid_type.get(t) == "intermediate":
                intermediate_edges[nid_to_id[t]].append(edge)

        for friendly_node, edges in intermediate_edges.items():
            print(f"\n🔍 替换中间节点: {friendly_node}（共 {len(edges)} 条边）")

            # 用于存储每个answer与中间节点的邻居交集结果
            mid_candidates_per_answer = defaultdict(set)

            # 标记当前中间节点是否连接了 answer 节点
            has_answer_node = False

            print(f"nid_to_id = {nid_to_id}")

            for edge in edges:
                src_nid = edge["start"]
                dst_nid = edge["end"]
                rel = edge["relation"]

                # 找到连接的另一个节点
                if nid_to_id.get(src_nid) == friendly_node:
                    other_nid = dst_nid
                else:
                    other_nid = src_nid
                other_type = nid_type[other_nid]

                if other_type == "answer":
                    has_answer_node = True
                    # 分别处理每个 answer
                    for ans in data["answer"]:
                        ans_mid = ans["answer_argument"]
                        neighbors = set(search_entity_neighbors(ans_mid, rel))
                        if not mid_candidates_per_answer[ans_mid]:
                            mid_candidates_per_answer[ans_mid] = neighbors
                        else:
                            mid_candidates_per_answer[ans_mid] &= neighbors
                else:
                    other_mid = nid_to_id[other_nid]
                    neighbors = set(search_entity_neighbors(other_mid, rel))
                    # 如果连接的是非answer节点，则需要和所有已存的answer分别计算交集
                    if has_answer_node:
                        for ans_mid in mid_candidates_per_answer:
                            mid_candidates_per_answer[ans_mid] &= neighbors
                    else:
                        # 另一侧也非answer节点，直接用 neighbors 初始化
                        if "single" not in mid_candidates_per_answer:
                            mid_candidates_per_answer["single"] = neighbors
                        else:
                            mid_candidates_per_answer["single"] &= neighbors

            # 现在分别处理每个 answer（或单个情况）的候选节点
            for ans_mid, candidate_mids in mid_candidates_per_answer.items():
                if not candidate_mids:
                    print(f"⚠️ 无共同实体替换 {friendly_node} 与 {ans_mid}")
                    continue
                print(f"✅ 中间节点 {friendly_node} 对于 {ans_mid} 候选实体为: {candidate_mids}")

                for mid in candidate_mids:
                    label = get_label_for_mids([mid]).get(mid, "Unknown")
                    entity_names[mid] = label
                    all_entities.add(mid)
                    explored_entities.add(mid)

                    # 初始化 graph 中新节点
                    if mid not in graph:
                        graph[mid] = {}

                    # 将 friendly node 的连接迁移到 mid 节点上
                    if friendly_node in graph:
                        for neighbor in graph[friendly_node]:
                            # ❗ 关键修正逻辑 ❗
                            # 如果neighbor是答案实体,只允许当前的ans_mid连接
                            if neighbor in mid_candidates_per_answer:
                                if neighbor != ans_mid:
                                    continue  # 跳过不属于当前mid的答案节点
                            for direction in ['forward', 'backward']:
                                for rel in graph[friendly_node][neighbor][direction]:
                                    # 正向迁移
                                    if neighbor not in graph[mid]:
                                        graph[mid][neighbor] = {'forward': set(), 'backward': set()}
                                    graph[mid][neighbor][direction].add(rel)

                                    # 反向迁移（更新邻居指向 mid）
                                    if mid not in graph[neighbor]:
                                        graph[neighbor][mid] = {'forward': set(), 'backward': set()}
                                    opposite_direction = 'backward' if direction == 'forward' else 'forward'
                                    graph[neighbor][mid][opposite_direction].add(rel)

            # 删除原始 friendly name 节点
            if friendly_node in graph:
                del graph[friendly_node]

            # 删除其他节点中指向 friendly_node 的信息
            for node in graph:
                if friendly_node in graph[node]:
                    del graph[node][friendly_node]

            # 从 all_entities 中移除
            if friendly_node in all_entities:
                all_entities.remove(friendly_node)

            # 可选：从 entity_names 中移除
            if friendly_node in entity_names:
                del entity_names[friendly_node]

        # Step 4: 如果某个 topic entity 没有邻居，则直接连接到所有答案
        for topic_id in topic_entity:
            if topic_id not in graph or not graph[topic_id]:
                print(f"⚠️ Topic entity {topic_id} 没有任何邻居，强制连接答案")

                # 先找出与该 topic entity 相关的边
                related_edges = []
                for edge in graph_query.get("edges", []):
                    s, t = edge["start"], edge["end"]
                    rel = edge["relation"]
                    if nid_to_id.get(s) == topic_id or nid_to_id.get(t) == topic_id:
                        related_edges.append(rel)

                # 如果找不到相关边，跳过（理论上不会发生）
                if not related_edges:
                    print(f"⚠️ 无法找到 topic entity {topic_id} 的相关关系边，跳过")
                    continue

                for answer in data["answer"]:
                    ans_mid = answer["answer_argument"]
                    ans_name = answer["entity_name"]

                    # 如果答案实体还不在图中，初始化
                    if ans_mid not in graph:
                        graph[ans_mid] = {}
                    if topic_id not in graph:
                        graph[topic_id] = {}

                    # 初始化双向连接
                    if ans_mid not in graph[topic_id]:
                        graph[topic_id][ans_mid] = {'forward': set(), 'backward': set()}
                    if topic_id not in graph[ans_mid]:
                        graph[ans_mid][topic_id] = {'forward': set(), 'backward': set()}

                    # 将所有相关边都用上（注意去重）
                    for rel in set(related_edges):
                        graph[topic_id][ans_mid]['forward'].add(rel)
                        graph[ans_mid][topic_id]['backward'].add(rel)

                    print(f"🔗 强连边: {topic_id} --{rel}--> {ans_mid} ({ans_name})")

    # ------------------------- cwq -------------------------
    elif dataset == "cwq":
        sparql_query = data.get("sparql", "")
        if not sparql_query:
            print("⚠️ No SPARQL query found in this CWQ item.")
            return graph, entity_names

        # 改写SPARQL查询，提取所有变量
        rewritten_sparql = force_select_all_vars(sparql_query)
        bindings = query_variable_bindings_all_vars(rewritten_sparql)

        # 收集所有的 mid
        all_mids = {mid for mids in bindings.values() for mid in mids}

        # 获取所有 mid 的 labels 和 types
        label_map = resolve_entity_labels(all_mids)
        type_map = resolve_entity_types(all_mids)

        # 更新 entity_names, explored_entities, all_entities
        for mid in all_mids:
            label = label_map.get(mid)
            if not label:
                types = type_map.get(mid, ["UNKNOWN"])
                label = types[0]
            entity_names[mid] = label

            if mid not in topic_entity:
                explored_entities.add(mid)
                all_entities.add(mid)

            if mid not in graph:
                graph[mid] = {}

        # 提取 SPARQL 中的 triples
        triple_lines = [
            line.strip()
            for line in sparql_query.split('\n')
            if 'ns:' in line and not line.strip().startswith(('PREFIX', 'FILTER', '#'))
        ]

        for triple in triple_lines:
            parts = triple.split()
            if len(parts) < 3:
                continue

            subj, rel, obj = parts[0], parts[1], parts[2]
            subj = subj.replace('ns:', '')
            obj = obj.replace('ns:', '')
            rel = rel.replace('ns:', '')

            # 处理变量 (以?开头) 的情况
            subj_mids = bindings.get(subj[1:], []) if subj.startswith('?') else [subj]
            obj_mids = bindings.get(obj[1:], []) if obj.startswith('?') else [obj]

            for subj_mid in subj_mids:
                for obj_mid in obj_mids:
                    # 确保节点存在于 graph
                    if subj_mid not in graph:
                        graph[subj_mid] = {}
                    if obj_mid not in graph:
                        graph[obj_mid] = {}

                    # 初始化 graph 中的连接结构
                    if obj_mid not in graph[subj_mid]:
                        graph[subj_mid][obj_mid] = {'forward': set(), 'backward': set()}
                    if subj_mid not in graph[obj_mid]:
                        graph[obj_mid][subj_mid] = {'forward': set(), 'backward': set()}

                    # 添加关系
                    graph[subj_mid][obj_mid]['forward'].add(rel)
                    graph[obj_mid][subj_mid]['backward'].add(rel)

                    print(f"➤ {subj_mid} ({entity_names.get(subj_mid, 'UNKNOWN')}) --{rel}--> {obj_mid} ({entity_names.get(obj_mid, 'UNKNOWN')})")

        # Step 额外修复: 处理直接出现在 triples 中但未在 bindings 中的 mid
        extra_mids_in_triples = set()
        mid_pattern = re.compile(r'ns:(m\.[a-zA-Z0-9_]+)')

        for triple in triple_lines:
            extra_mids_in_triples.update(mid_pattern.findall(triple))

        # 仅处理尚未被处理的 mids
        unprocessed_mids = extra_mids_in_triples - all_mids - set(topic_entity.keys())
        if unprocessed_mids:
            extra_labels = resolve_entity_labels(unprocessed_mids)
            extra_types = resolve_entity_types(unprocessed_mids)

            for mid in unprocessed_mids:
                label = extra_labels.get(mid)
                if not label:
                    types = extra_types.get(mid, ["UNKNOWN"])
                    label = types[0]

                entity_names[mid] = label
                explored_entities.add(mid)
                all_entities.add(mid)

                if mid not in graph:
                    graph[mid] = {}

                print(f"✅ 补充添加未处理的中间实体: {mid} ({label})")

    # ------------------------- WebQSP -------------------------
    elif dataset == "webqsp":
        parses = data.get("Parses", [])
        if not parses:
            print("⚠️ No Parses found in this WebQSP item.")
            return graph, entity_names

        for parse in parses:
            sparql_query = parse.get("Sparql", "")
            if not sparql_query:
                print("⚠️ No SPARQL query in this parse.")
                continue

            # 改写SPARQL查询，提取所有变量
            rewritten_sparql = force_select_all_vars(sparql_query)
            bindings = query_variable_bindings_all_vars(rewritten_sparql)

            # 收集所有的 mid
            all_mids = {mid for mids in bindings.values() for mid in mids}

            # 获取所有 mid 的 labels 和 types
            label_map = resolve_entity_labels(all_mids)
            type_map = resolve_entity_types(all_mids)

            # 更新 entity_names, explored_entities, all_entities
            for mid in all_mids:
                label = label_map.get(mid)
                if not label:
                    types = type_map.get(mid, ["UNKNOWN"])
                    label = types[0]
                entity_names[mid] = label

                if mid not in topic_entity:
                    explored_entities.add(mid)
                    all_entities.add(mid)

                if mid not in graph:
                    graph[mid] = {}

            # 提取 SPARQL 中的 triples
            triple_lines = [
                line.strip()
                for line in sparql_query.split('\n')
                if 'ns:' in line and not line.strip().startswith(('PREFIX', 'FILTER', '#'))
            ]

            for triple in triple_lines:
                parts = triple.split()
                if len(parts) < 3:
                    continue

                subj, rel, obj = parts[0], parts[1], parts[2]
                subj = subj.replace('ns:', '')
                obj = obj.replace('ns:', '')
                rel = rel.replace('ns:', '')

                # 处理变量 (以?开头) 的情况
                subj_mids = bindings.get(subj[1:], []) if subj.startswith('?') else [subj]
                obj_mids = bindings.get(obj[1:], []) if obj.startswith('?') else [obj]

                for subj_mid in subj_mids:
                    for obj_mid in obj_mids:
                        # 确保节点存在于 graph
                        if subj_mid not in graph:
                            graph[subj_mid] = {}
                        if obj_mid not in graph:
                            graph[obj_mid] = {}

                        # 初始化 graph 中的连接结构
                        if obj_mid not in graph[subj_mid]:
                            graph[subj_mid][obj_mid] = {'forward': set(), 'backward': set()}
                        if subj_mid not in graph[obj_mid]:
                            graph[obj_mid][subj_mid] = {'forward': set(), 'backward': set()}

                        # 添加关系
                        graph[subj_mid][obj_mid]['forward'].add(rel)
                        graph[obj_mid][subj_mid]['backward'].add(rel)

                        print(f"➤ {subj_mid} ({entity_names.get(subj_mid, 'UNKNOWN')}) --{rel}--> {obj_mid} ({entity_names.get(obj_mid, 'UNKNOWN')})")

            # Step 额外修复: 处理直接出现在 triples 中但未在 bindings 中的 mid
            extra_mids_in_triples = set()
            mid_pattern = re.compile(r'ns:(m\.[a-zA-Z0-9_]+)')

            for triple in triple_lines:
                extra_mids_in_triples.update(mid_pattern.findall(triple))

            # 仅处理尚未被处理的 mids
            unprocessed_mids = extra_mids_in_triples - all_mids - set(topic_entity.keys())
            if unprocessed_mids:
                extra_labels = resolve_entity_labels(unprocessed_mids)
                extra_types = resolve_entity_types(unprocessed_mids)

                for mid in unprocessed_mids:
                    label = extra_labels.get(mid)
                    if not label:
                        types = extra_types.get(mid, ["UNKNOWN"])
                        label = types[0]

                    entity_names[mid] = label
                    explored_entities.add(mid)
                    all_entities.add(mid)

                    if mid not in graph:
                        graph[mid] = {}

                    print(f"✅ 补充添加未处理的中间实体: {mid} ({label})")

    gt_entities = all_entities

    print(f"topic_entity = {topic_entity}")
    print(f"real_answers = {question_real_answer}")
    print(f"entity_names = {entity_names}")
    print(f"explored_entities = {explored_entities}")
    print(f"next_explore_entities = {next_explore_entities}")
    print(f"all_entities = {all_entities}")
    print(f"ground truth entities = {gt_entities}")

    graph = {k: v for k, v in graph.items() if k.startswith("m.")}

    print(f"\n开始添加非 ground-truth 部分:)\n")
    for _ in range(depth):
        graph, all_entities, explored_entities, next_explore_entities, entity_names = explore_graph_from_one_topic_entities(
            next_explore_entities,
            graph,
            entity_names,
            explored_entities,
            all_entities,
            limit_per_entity=100 if dataset in {"simpleqa", "webquestions"} else 10
        )
    print(f"all_entities = {all_entities}")

    return graph, entity_names, gt_entities



