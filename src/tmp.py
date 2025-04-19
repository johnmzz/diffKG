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
from collections import deque
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
def explore_graph_from_one_topic_entities(current_entities, graph, entity_names, exlored_entities, all_entities, answer_name=[]):
    """
    current_entities: set(str), 当前轮次中要扩展的实体集合（Freebase MID）
    graph: dict, 当前正在构建的图结构
    entity_names: dict, MID → 实体名称
    exlored_entities: set(str), 已所有已探索过的实体，防止重复
    all_entities: set(str), 所有发现的实体（全局集合）
    answer_name: list(str), 正确答案的实体名称
    """
    storage_lock = Lock()  # 为多线程访问共享数据结构（如 graph）设置的线程锁。
    new_entities = set()    # 当前轮新发现的实体集合
    answer_name_set = set(answer_name)
    found_answer_entities = set()
    
    # 多线程并发：为每个实体提交一个异步任务，查询它的一跳边
    with ThreadPoolExecutor(max_workers=8) as executor:    # 创建线程池
        futures = {executor.submit(search_relations_and_entities_combined_1, entity): entity for entity in current_entities}

        exlored_entities.update(current_entities)   # 将 current_entities 加入 exlored_entities 标记为已处理。

        # 处理每个异步任务的结果（一个实体的所有边）
        for future in as_completed(futures):    # as_completed(futures) waits for queries to finish in any order.
            results = future.result()   # get query result
            entity = futures[future]    # retrieve corresponding entity of the finished "future"

            # 处理每条边结果
            for result in results:      # eg. a single result (one edge in KG) = { "relation": "shares_border_with", "connectedEntity": "m.0f8l9c", "connectedEntityName": "France", "direction": "tail" }
                relation, connected_entity, connected_name, direction = result['relation'], result['connectedEntity'], result['connectedEntityName'], result['direction']

                if connected_entity.startswith("m."):
                    if connected_entity in exlored_entities:    # skip already explored entities
                        continue

                    # 检查是否匹配到答案名称
                    if connected_name in answer_name_set:
                        found_answer_entities.add(connected_entity)

                    with storage_lock:      # lock when modifying shared structures
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

        # 更新新实体集合 & 下一轮准备
        new_entities.difference_update(exlored_entities)    # remove explored entities from new_entities
        all_entities.update(new_entities)                   # add new entities to the global entity set
        current_entities = new_entities                     # set new entities as "current_entities" for next round
        
        # 判断是否找到左右答案 entity
        found_all_answers = len(found_answer_entities) == len(answer_name_set)

    # print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (graph, all_entities, exlored_entities, current_entities, entity_names, found_all_answers)



# 从一个实体出发，在 Freebase 知识图谱中查询与之连接的所有三元组，并返回清洗后的结构化结果。
def search_relations_and_entities_combined_1(entity_id):

    # 查询某个实体与其连接的所有边（关系）与相邻实体，并返回它们的名称和连接方向。
    sparql_query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?relation ?connectedEntity ?connectedEntityName ?direction
    WHERE {
        {
            ns:%s ?relation ?connectedEntity .
            OPTIONAL {
                ?connectedEntity ns:type.object.name ?name .
                FILTER(lang(?name) = 'en')
            }
            BIND(COALESCE(?name, "Unnamed Entity") AS ?connectedEntityName)
            BIND("tail" AS ?direction)
        }
        UNION
        {
            ?connectedEntity ?relation ns:%s .
            OPTIONAL {
                ?connectedEntity ns:type.object.name ?name .
                FILTER(lang(?name) = 'en')
            }
            BIND(COALESCE(?name, "Unnamed Entity") AS ?connectedEntityName)
            BIND("head" AS ?direction)
        }
    }
    """ % (entity_id, entity_id)

    # Execute the SPARQL Query
    results = execute_sparql(sparql_query)  # sends the query to a SPARQL endpoint, retrieves JSON results.

    return replace_prefix1(results)



# 将 http://rdf.freebase.com/ns/ 前缀清洗掉，保留 m.02jx3 这样的简写 MID。
def replace_prefix1(data):
    if data is None:
        print("Warning: No data available to process in replace_prefix1.")
        return []

    # Returns a list of cleaned dictionaries.
    return [{key: value['value'].replace("http://rdf.freebase.com/ns/", "") for key, value in result.items()} for result in data]



# Sends a SPARQL query to a Freebase/Wikidata endpoint and retrieves structured data in JSON format.
#   Returns a list of dictionaries (extracted from "bindings") or None if the query fails.
def execute_sparql(sparql_txt):

    # Assuming SPARQLPATH is a variable that holds your SPARQL endpoint URL
    sparql = SPARQLWrapper(SPARQLPATH)      # Creates a SPARQL query wrapper
    # sparql.setQuery(sparql_txt)
    # sparql.setQuery("define sql:big-data-const 0\n" + sparql_txt)
    sparql.setQuery("define sql:big-data-const 0\n" + sparql_txt + "\nLIMIT 3000")
    sparql.setReturnFormat(JSON)
    
    attempts = 0
    while attempts < 3:  # Set the number of retries
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print("404 Error encountered. Retrying after 2 seconds...")
            print(e)

            time.sleep(2)  # Sleep for 2 seconds before retrying
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



def construct_subgraph(dataset, data, topic_entity, entity_names, question_real_answer, depth):
    print("--------- Construct Subgraph ---------")
    explored_entities = set()                           # 已经被访问过的实体
    next_explore_entities = set(topic_entity.keys())    # 当前这轮 BFS 将要扩展的实体（初始为 topic entities）
    all_entities = set(topic_entity.keys())             # 记录图中所有访问到的实体

    # 临时维护的子图（邻接表结构）
    graph = {entity: {} for entity in topic_entity.keys()}

    # ------------------------- SimpleQA -------------------------
    if dataset == "simpleqa":
        # ...
        pass

    # ------------------------- WebQuestions -------------------------
    elif dataset == "webquestions":
        # ...
        pass

    # ------------------------- GrailQA -------------------------
    elif dataset == "grailqa":
        # ...
        pass

    # ------------------------- cwq -------------------------
    elif dataset == "cwq":
        sparql = data.get("sparql", "")
        if not sparql:
            print("⚠️ 无 SPARQL 查询语句，跳过")
            return graph, entity_names

        # 获取所有实体三元组（不包含 FILTER/PREFIX）
        triple_lines = [
            line.strip() for line in sparql.split('\n')
            if 'ns:' in line and not line.strip().startswith(('PREFIX', 'FILTER', '#'))
        ]

        # 提取所有 mid
        mids_in_triples = extract_mids(triple_lines)

        # 解析 mid 的 label
        label_map = resolve_entity_labels(mids_in_triples)

        # 构造图结构
        for line in triple_lines:
            parts = re.findall(r'ns:([^\s]+)', line)
            if len(parts) == 3:
                h_raw, rel_raw, t_raw = parts
                h = h_raw if h_raw.startswith("m.") else f"?{h_raw}"
                t = t_raw if t_raw.startswith("m.") else f"?{t_raw}"
                rel = rel_raw

                # 如果是实体（mid），则初始化图节点
                for node in [h, t]:
                    if node.startswith("m."):
                        if node not in graph:
                            graph[node] = {}
                        all_entities.add(node)
                        if node not in entity_names:
                            entity_names[node] = label_map.get(node, "UNKNOWN")

                # 建图（仅实体对）
                if h.startswith("m.") and t.startswith("m."):
                    if t not in graph[h]:
                        graph[h][t] = {'forward': set(), 'backward': set()}
                    if h not in graph[t]:
                        graph[t][h] = {'forward': set(), 'backward': set()}
                    graph[h][t]['forward'].add(rel)
                    graph[t][h]['backward'].add(rel)
                    print(f"➤ {h} ({entity_names[h]}) --{rel}--> {t} ({entity_names[t]})")

        # 将答案实体加入 explored/all/entities
        for ans_mid in question_real_answer:
            entity_names[ans_mid] = get_label_for_mids([ans_mid]).get(ans_mid, "UNKNOWN")
            all_entities.add(ans_mid)
            explored_entities.add(ans_mid)
            if ans_mid not in graph:
                graph[ans_mid] = {}

        # 记录所有 topic_entity
        for mid in topic_entity:
            explored_entities.add(mid)

        print(f"✅ 构图完成，共实体: {len(all_entities)}, 边数: {sum(len(v) for v in graph.values())}")
