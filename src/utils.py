from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import json
from texttable import Texttable
from functools import lru_cache
import re
from SPARQLWrapper import SPARQLWrapper, JSON, XML
import time
from urllib.error import HTTPError
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed


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

    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   

    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        ID = 'question'
        question_string = 'question'

    # elif dataset_name == 'trex':
    #     with open('../data/T-REX.json',encoding='utf-8') as f:
    #         datas = json.load(f)
    #     question_string = 'input'  

    # elif dataset_name == 'zeroshotre':
    #     with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
    #         datas = json.load(f)
    #     question_string = 'input'

    # elif dataset_name == 'creak':
    #     with open('../data/creak.json',encoding='utf-8') as f:
    #         datas = json.load(f)
    #     question_string = 'sentence'
    #     ID = 'ex_id'

    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string, ID


def check_answerlist(dataset_name, question_string, ori_question, ground_truth_datas, origin_data):
    answer_list= []
    # origin_data = [j for j in ground_truth_datas if j[question_string] == ori_question]
    if dataset_name == 'cwq':
        answer_list.append(origin_data["answer"])

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]

    return list(set(answer_list))


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


# Interacts with either Google's Gemini API or OpenAI's GPT API to generate responses based on a given prompt
# temperature controls randomness: lower values (0.1-0.4) more deterministic, higher values (0.7-1.0) more creative
def run_LLM(prompt, model,temperature=0.4):
    result = ''
    if "google" in model:
        genai.configure(api_key="your_api_keys")

        # model = genai.GenerativeModel('gemini-1.5-flash')
        model = genai.GenerativeModel("gemini-1.5-flash")
        system_message = "You are an AI assistant that helps people find information."

        chat = model.start_chat(
            history=[
                {"role": "user", "parts": system_message},
            ]
        )

        try_time = 0
        while try_time<3:
            try:
                response = chat.send_message(prompt)
                print("Google response: ")
                return (response.text)
                break
            except Exception as e:
                error_message = str(e)
                print(f"Google error: {error_message}")
                print("Retrying in 2 seconds...")
                try_time += 1
                time.sleep(40)
                    

    # openai_api_base = "http://localhost:8000/v1"
    elif "gpt" in model:
        openai_api_key = "your_api_keys"
        if model == "gpt4":
            # model = "gpt-4-0613"
            model = "gpt-4-turbo"
        else:
            model = "gpt-3.5-turbo-0125"
        # model = "gpt-3.5-turbo-0125"
        # model = "gpt-3.5-turbo"
        # model = "gpt-4-turbo"
        # model = "gpt-4o"

        # create an OpenAI client for sending request
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            # base_url=openai_api_base,
        )

    else:   # self-hosted local LLM
        openai_api_base = "http://localhost:8000/v1"
        openai_api_key = "EMPTY"
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    
    # construct OpenAI message
    system_message = "You are an AI assistant that helps people find information."
    messages = [{"role": "system", "content": system_message}]      # system messsage: sets the AI's role
    message_prompt = {"role": "user", "content": prompt}            # user prompt: contains the actual query
    messages.append(message_prompt)

    # Sending OpenAI API Request: calls OpenAI's chat API, retries up to 3 times if an error occurs.
    try_time = 0
    while try_time<3:
        try:
            response = client.chat.completions.create(
                model=model,                # GPT-4 Turbo or GPT-3.5 Turbo
                messages=messages,
                temperature=temperature,
                max_tokens=512,             # limits response length
                frequency_penalty= 0,       # avoid repetition
                presence_penalty=0          # no penalty for introducing new topics
            )
            result = response.choices[0].message.content    # extract the generated response
            break
        except Exception as e:
            error_message = str(e)
            print(f"OpenAI error: {error_message}")
            print("Retrying in 2 seconds...")
            try_time += 1
            time.sleep(2)

    print(f"{model} response: ")

        # print("end openai")

    return result


# Retrieves the name or type of an entity given its entity ID by querying a SPARQL knowledge graph (KG) endpoint.
#   Uses an LRU (Least Recently Used) caches results for up to 1,024 different entity queries for fast retrieval.
#       eg. id2entity_name_or_type("m.02jx3")  # First call → Runs SPARQL query
#           id2entity_name_or_type("m.02jx3")  # Second call → Uses cached result
#   Executes a SPARQL query to get the human-readable name of an entity.
@lru_cache(maxsize=1024)
def id2entity_name_or_type(entity_id):
    init_id = entity_id

    # Formats the SPARQL query string, replacing placeholders with entity_id
    entity_id = sparql_id % (format(entity_id), format(entity_id))

    # prepare SPARQL query
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(entity_id)
    sparql.setReturnFormat(JSON)
    # results = sparql.query().convert()

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

    # Process query result
    if len(results["results"]["bindings"]) == 0:    # if no results are found, return "Unnamed Entity"
        return "Unnamed Entity"
    else:
        # Extract entity name
        #   First, filter to find results with 'xml:lang': 'en'
        english_results = [result['tailEntity']['value'] for result in results["results"]["bindings"] if result['tailEntity'].get('xml:lang') == 'en']
        if english_results:
            return english_results[0]  # Return the first English result

        #   If no English labels are found, checks for names that contain only letters and numbers (ignores symbols)
        alphanumeric_results = [result['tailEntity']['value'] for result in results["results"]["bindings"]
                                if re.match("^[a-zA-Z0-9 ]+$", result['tailEntity']['value'])]
        if alphanumeric_results:
            return alphanumeric_results[0]  # Return the first alphanumeric result
        
        # If no English or alphanumeric names are found, returns "Unnamed Entity".
        return "Unnamed Entity"


# 从一个实体集合出发，不断向外扩展，构建包含实体与关系的邻接表，并避免重复探索。
# current_entities: set(str), 当前轮需要扩展的实体集合（Freebase MID）
# graph: dict, 构建中的图
# entity_names: dict, 每个实体 MID 对应的实体名称
# exlored_entities: set(str), 已探索过的实体集合，防止重复处理。
# all_entities: set(str), 总共发现过的所有实体集合。
def explore_graph_from_one_topic_entities(current_entities, graph, entity_names, exlored_entities, all_entities):

    storage_lock = Lock()  # 为多线程访问共享数据结构（如 graph）设置的线程锁。

    print(f"Exploring entities ...")
    start = time.time()
    new_entities = set()    # 当前轮新发现的实体集合
    
    # One thread for each entity, for single-entity questions, multi-threading will be needed for 2nd+ rounds.
    with ThreadPoolExecutor(max_workers=80) as executor:    # create thread pool

        # Creates a dictionary where for all entities to process (to track which "future" associates which entity):
        #   Keys -> Future objects (asynchronous tasks: run search_relations_and_entities_combined_1(entity)).
        #   Values -> The corresponding entity being processed.
        #   eg.
        #   {
        #       <Future at 0x1234567890 state=running>: "m.02jx3",  # Germany
        #       <Future at 0x1234567891 state=running>: "m.03q5f",  # Harvard University
        #       <Future at 0x1234567892 state=running>: "m.0f8l9c"   # France
        #   }
        futures = {executor.submit(search_relations_and_entities_combined_1, entity): entity for entity in current_entities}

        exlored_entities.update(current_entities)   # 将 current_entities 加入 exlored_entities 标记为已处理。

        # 异步处理任务结果
        for future in as_completed(futures):    # as_completed(futures) waits for queries to finish in any order.
            results = future.result()   # get query result
            entity = futures[future]    # retrieve corresponding entity of the finished "future"

            # 处理每条边结果
            for result in results:      # eg. a single result (one edge in KG) = { "relation": "shares_border_with", "connectedEntity": "m.0f8l9c", "connectedEntityName": "France", "direction": "tail" }
                relation, connected_entity, connected_name, direction = result['relation'], result['connectedEntity'], result['connectedEntityName'], result['direction']

                if connected_entity.startswith("m."):
                    if connected_entity in exlored_entities:    # skip already explored entities
                        continue

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
        # print ((all_entities))
        # print((exlored_entities))
        # print ((current_entities))

    # print("Entities are not fully connected or answer entity not found within the maximum allowed hops.")
    return (graph, all_entities, exlored_entities, current_entities, entity_names)


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
    # eg. Query Entity: m.02jx3 "Germany"
    #
    # |       Relation	     | Connected Entity |    Name	 |  Direction |
    # | "shares_border_with" |    "m.0f8l9c"	|  "France"  |	 "tail"   |
    # | "located_in"	     |    "m.0g7qf"	    |  "Europe"  |	 "tail"   |
    # | "shares_border_with" |    "m.0g7qf"	    |  "Belgium" |	 "head"   |

    # Execute the SPARQL Query
    results = execute_sparql(sparql_query)  # sends the query to a SPARQL endpoint, retrieves JSON results.

    # eg. returns (after execute_sparql and replace_prefix1):
    # [
    #     { "relation": "shares_border_with", "connectedEntity": "m.0f8l9c", "connectedEntityName": "France", "direction": "tail" },
    #     { "relation": "located_in", "connectedEntity": "m.0g7qf", "connectedEntityName": "Europe", "direction": "tail" }
    # ]

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
    sparql.setQuery(sparql_txt)
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
    # eg. raw qury
    # {
    # "head": { "vars": ["relation", "connectedEntity", "connectedEntityName", "direction"] },
    # "results": {
    #     "bindings": [
    #     {
    #         "relation": { "value": "http://rdf.freebase.com/ns/shares_border_with" },
    #         "connectedEntity": { "value": "http://rdf.freebase.com/ns/m.0f8l9c" },
    #         "connectedEntityName": { "value": "France" },
    #         "direction": { "value": "tail" }
    #     },
    #     {
    #         "relation": { "value": "http://rdf.freebase.com/ns/located_in" },
    #         "connectedEntity": { "value": "http://rdf.freebase.com/ns/m.0g7qf" },
    #         "connectedEntityName": { "value": "Europe" },
    #         "direction": { "value": "tail" }
    #     }
    #     ]
    # }
    # }