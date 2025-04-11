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
        self.load_data_time = 0.0
        self.to_torch_time = 0.0
        self.num_LLM_call = 0
        self.results = []           # 保存训练/测试时输出的结果（比如 loss、MAE、准确率等）

        # 设置计算设备
        self.use_gpu = torch.cuda.is_available()
        print("use_gpu = ", self.use_gpu)
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        # 加载数据
        self.load_data()

    

    def load_data(self):
        t1 = time.time()
        file_name = self.args.dataset

        subgraph_db = f'subgraph/{file_name}_main_Subgraphs.db'     # 存储 subgraph
        path_db = f'subgraph/{file_name}_path.db'                   # 存储 path
        answer_db = f'answer/{file_name}_answer.db'

        # 加载数据集：
        # datas: 数据集（JSON）
        # question_string: 用于提取文字问题的 key
        # Q_id: 用于提取 ID 的 key
        datas, question_string, Q_id = prepare_dataset(file_name)

        for data in datas[0:1]:
            # 提取数据
            question = data[question_string]        # 文本问题 (str)
            topic_entity = data['topic_entity']     # topic entities (dict)
            question_id = data[Q_id]                # 问题 ID (str)

            # 提取 ground-truth 答案 (list)
            question_real_answer = check_answerlist(file_name, question_string, question, datas, data)

            print(f"question: {question}")
            print(f"answers = {question_real_answer}")
            print(topic_entity.values())
            print("list(topic_entity):", list(topic_entity.items()))

            # LLM prompt
            # prompt_split = split_question_prompt + "\nQ:\n Question: " + question + "?"+ "\n"
            # prompt_split += "Main Topic Entities: \n" + str(data['topic_entity']) + "\n" +"A:\n"

            # split_answer = run_LLM(prompt_split, LLM_model)[2:]
            # self.num_LLM_call += 1

            # predict_length, thinking_cot_line = extract_path_length_from_text(split_answer)
            # split_question = extract_split_questions(split_answer)
            # print("predict CoT length:", predict_length)

            # oder entities based on CoT (not needed?)
            # sorted_topic_entity_name = reorder_entities(thinking_cot_line, list(topic_entity.values()))

            entity_names = {entity_id: id2entity_name_or_type(entity_id) for entity_id in topic_entity}

            # entity tracking sets
            explored_entities = set()
            next_explore_entities = set(topic_entity.keys())
            all_entities = set(topic_entity.keys())

            print(next_explore_entities)
            print(all_entities)

            # create an empty adj-list for KG, each entity is a key, and its value is an empty dictionary ({}) that will store connections.
            # eg.
            # graph = {
            #     "m.02jx3": {  # Germany
            #         "m.0f8l9c": {  # France
            #             "forward": {"shares_border_with"},
            #             "backward": set()
            #         },
            #         "m.0g7qf": {  # Europe
            #             "forward": {"located_in"},
            #             "backward": set()
            #         }
            #     },
            #     "m.0f8l9c": {  # France
            #         "m.02jx3": {
            #             "forward": set(),
            #             "backward": {"shares_border_with"}
            #         },
            #         "m.0g7qf": {  # Europe
            #             "forward": {"part_of"},
            #             "backward": set()
            #         }
            #     },
            #     ...
            # }
            graph = {entity: {} for entity in topic_entity.keys()}

            # 提取 sub-graph

            for entity_id in topic_entity:
                # all_entities: list(eid), 
                graph, all_entities, explored_entities, next_explore_entities, entity_names = explore_graph_from_one_topic_entities(
                    next_explore_entities, graph, entity_names, explored_entities, all_entities)

            # 保存到文件（美化缩进）
            graph_json_ready = convert_graph_sets_to_lists(graph)
            with open("graph.json", "w", encoding="utf-8") as f:
                json.dump(graph_json_ready, f, indent=4)
                json.dump(entity_names, f, indent=4)

            print(f"all entities = {all_entities}")
            print(f"explored entities = {explored_entities}")
            print(f"next explore entities = {next_explore_entities}")
            print(f"entity_names = {entity_names}")