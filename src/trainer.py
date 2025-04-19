import sys
import time
from typing import List
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp
import torch_geometric as pyg
import networkx as nx
import operator
import json
from utils import prepare_dataset, check_answerlist, id2entity_name_or_type, explore_graph_from_one_topic_entities
from utils import init_LLM, run_LLM, extract_path_length_from_text, extract_split_questions, reorder_entities, explore_graph_from_entities_by_hop_neighbor_1
from cot_prompt_list import split_question_prompt
from utils import initialize_large_database, delete_data_by_question_id, save_to_large_db, load_from_large_db, construct_subgraph

def convert_graph_sets_to_lists(graph):
    return {
        str(src): {
            str(dst): {
                "forward": list(rels["forward"]),
                "backward": list(rels["backward"])
            } for dst, rels in nbrs.items()
        } for src, nbrs in graph.items()
    }

class DataSet:
    def __init__(self, id, qtype, question, entities, answer):
        self.id = id
        self.qtype = qtype
        self.question = question
        self.entities = entities
        self.answer = answer

class Trainer(object):
    def __init__(self, args):
        self.args = args            # 保存参数对象，供其他方法使用。
        self.num_questions = 0
        self.num_not_found = 0
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.num_LLM_call = 0
        self.num_invalid_questions = 0
        self.results = []           # 保存训练/测试时输出的结果（比如 loss、MAE、准确率等）

        # 设置计算设备
        self.use_gpu = torch.cuda.is_available()
        print("use_gpu = ", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        # 设置 LLM
        if (self.args.llm == "openchat"):
            self.llm = "openchat/openchat-3.5-0106"
        elif (self.args.llm == "mistralai7B"):
            self.llm = "mistralai/Mistral-7B-Instruct-v0.3"
        elif (self.args.llm == "nousResearch"):
            self.llm = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
        elif (self.args.llm == "mixtral8x"):
            self.llm = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif (self.args.llm == "llama2"):
            self.llm = "meta-llama/Llama-2-13b-chat-hf"
        elif (self.args.llm == "llama3"):
            self.llm = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif (self.args.llm == "gemma"):
            self.llm = "google/gemma-7b-it"
        elif (self.args.llm == "zephyr"):
            self.llm = "HuggingFaceH4/zephyr-7b-beta"
        elif (self.args.llm == "starling"):
            self.llm = "berkeley-nest/Starling-LM-7B-alpha"
        elif (self.args.llm == "qwen"):
            self.llm = "Qwen/Qwen1.5-7B-Chat"
        elif (self.args.llm == "yi"):
            self.llm = "01-ai/Yi-1.5-9B-Chat"

        # 设置 sub-graph 提取 hop 数
        if (self.args.dataset == "simpleqa"):
            self.depth = 1
        elif (self.args.dataset == "webquestions"):
            self.depth = 1
        elif (self.args.dataset == "grailqa"):
            self.depth = 2
        elif (self.args.dataset == "webqsp"):
            self.depth = 2
        elif (self.args.dataset == "cwq"):
            self.depth = 2

        # 加载数据
        self.load_data()



    def load_data(self):
        t1 = time.time()
        file_name = self.args.dataset

        subgraph_db = f'/data1/zhuom/subgraph/{file_name}_main_Subgraphs.db'     # 存储 subgraph
        path_db = f'/data1/zhuom/subgraph/{file_name}_path.db'                   # 存储 path
        answer_db = f'/data1/zhuom/answer/{file_name}_answer.db'                 # 存储答案

        initialize_large_database(subgraph_db)

        # 加载数据集：
        # datas: 数据集（JSON）
        # question_string: 用于提取文字问题的 key
        # Q_id: 用于提取 ID 的 key
        datas, question_string, Q_id = prepare_dataset(file_name)

        for i, data in enumerate(datas[0:1]):
            print(f"提取数据 {i}")
            question = data[question_string]        # 文本问题 (str)
            topic_entity = data['topic_entity']     # topic entities (dict)
            question_id = data[Q_id]                # 问题 ID (str)
            entity_names = {entity_id: id2entity_name_or_type(entity_id) for entity_id in topic_entity}
            
            print(f"question: {question}")
            print("list(topic_entity):", list(topic_entity.items()))
            print(f"查询后的 entity names = {list(entity_names.items())}")

            # self.llm_model, self.tokenizer = init_LLM(self.llm)

            # # LLM prompt
            # prompt_split = split_question_prompt + "\nQ:\n Question: " + question + "?"+ "\n"
            # prompt_split += "Main Topic Entities: \n" + str(data['topic_entity']) + "\n" +"A:\n"

            # split_answer = run_LLM(prompt_split, self.args.llm, self.llm_model, self.tokenizer)[2:]
            # self.num_LLM_call += 1

            # predict_length, thinking_cot_line = extract_path_length_from_text(split_answer)
            # split_question = extract_split_questions(split_answer)

            # print(f"\nCoT = {thinking_cot_line}")
            # print(f"predicted length = {predict_length}")
            # print(f"split_questions = {split_question}")

            # # 根据 CoT 来对 topic entity 名称排序
            # sorted_topic_entity_name = reorder_entities(thinking_cot_line, list(topic_entity.values()))
            # # 将排序后的实体名字重新映射回实体 ID，供后续图探索使用
            # sorted_topic_entity_id = []
            # for name in sorted_topic_entity_name:
            #     for id, entity in topic_entity.items():
            #         if entity == name:
            #             sorted_topic_entity_id.append(id)
            #             break

            # print(f"sorted topic entities = {sorted_topic_entity_name}")
            # print(f"sorted topic entity IDs = {sorted_topic_entity_id}")

            # 提取 sub-graph
            subgraph_dict = load_from_large_db(subgraph_db, question_id)

            if subgraph_dict:
                print("Database 中找到问题: {question_id}，直接提取")
                graph = subgraph_dict['subgraph']
                entity_names = subgraph_dict['topic_entity']
                all_entities = subgraph_dict['all_entities']
                hops = subgraph_dict['hop']
                found_answers = subgraph_dict['found_answers']
                question_real_answer = subgraph_dict['answers']
                
                self.num_questions += 1

                print(f"answers = {question_real_answer}")
            else:
                print("Database 中未找到问题: {question_id}，需要从 KG 中提取")
                # 提取 ground-truth 答案 (list)
                question_real_answer = check_answerlist(file_name, data, topic_entity)

                if not question_real_answer:
                    self.num_not_found += 1
                    print(f"⚠️ Skipping QID {question_id} - no valid answers")
                    continue
                self.num_questions += 1
                print(f"\nanswers = {question_real_answer}")

                graph, all_entities, gt_entities = construct_subgraph(file_name, data, topic_entity, entity_names, question_real_answer, self.depth)
                # 保存到文件（美化缩进）
                graph_json_ready = convert_graph_sets_to_lists(graph)
                with open("graph.json", "w", encoding="utf-8") as f:
                    json.dump(graph_json_ready, f, indent=4)
                    json.dump(entity_names, f, indent=4)

                continue

                subgraph_dict = {
                    "question": question,
                    "question_id": question_id,
                    "answers": question_real_answer,
                    "topic_entity": entity_names,
                    "hop": hops,
                    "subgraph": graph,
                    "all_entities": all_entities,
                    "found_answers": found_answers,
                    "connected": True
                }
                delete_data_by_question_id(subgraph_db, question_id)
                save_to_large_db(subgraph_db, question_id, subgraph_dict)

        print(f"\n找到所有答案的问题 = {self.num_questions}")
        print(f"未找到所有答案的问题 = {self.num_not_found}")