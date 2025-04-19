"""Getting params from the command line."""

import argparse

# 从命令行读取参数配置，为 main.py 和 trainer.py 等模块提供运行时控制。
def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run DiffGED.")
    
    # Top-k 匹配方式（并行或顺序），默认 parallel
    parser.add_argument('--topk-approach', choices=['parallel','sequential'],default='parallel', help="Choose a top-k mapping generation approach: parallel, or sequential.")
    
    # 推理时取 Top-k 的 k 值，默认 100
    parser.add_argument('--test-k', type=int, default=100,help='Set k for inference.')
    
    # 用于 Top-k 分析的范围
    parser.add_argument('--k-range', type=list, default=[1,10,20,30,40,50,60,70,80,90,100],help='range of k for top-k approach analysis.')
    
    # 实验类型：测试、Top-k 分析、多样性分析
    parser.add_argument('--experiment', choices=['test', 'topk_analysis', 'diversity_analysis'],default='test', help="Choose an experiment: test, topk_analysis, or diversity_analysis.")
    
    # 使用的测试数据集：test/small/large
    parser.add_argument('--testset', choices=['test', 'small', 'large'],default='test', help="Choose a testing graph set: test, small, or large.")


    ################################ Diffusion 参数 ################################
    # 训练时使用的 diffusion 步数
    parser.add_argument('--diffusion-steps', type=int, default=1000)

    # 测试或推理时使用的 diffusion 步数（加速）
    parser.add_argument('--inference-diffusion_steps', type=int, default=10)


    ################################ 模型结构参数 ################################
    # 每一层神经网络的维度
    parser.add_argument("--hidden-dim",
                        type=list,
                        default=[128,64,32,32,32,32],
	                help="List of hidden dimensions.")

    # 每个 batch 包含的图对数量
    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    # 学习率
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    # L2 正则项（Adam 的权重衰减）
    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")


    ################################ 路径参数 ################################
    # 项目根目录（可作为数据或模型加载的基路径）
    parser.add_argument("--abs-path",
                        type=str,
                        default="../",
                        help="the absolute path")

    # 评估结果保存路径
    parser.add_argument("--result-path",
                        type=str,
                        default='result/',
                        help="Where to save the evaluation results")
    
    # 训练好的模型保存路径
    parser.add_argument("--model-path",
                        type=str,
                        default='model_save/',
                        help="Where to save the trained model")


    ################################ 模型训练控制 ################################
    # 是否执行训练（1 为训练，0 为只测试）
    parser.add_argument("--model-train",
                        type=int,
                        default=1,
                        help='Whether to train the model')

    # 从第几个 epoch 开始训练（可用于断点恢复）
    parser.add_argument("--model-epoch-start",
                        type=int,
                        default=0,
                        help="The number of epochs the initial saved model has been trained.")

    # 训练多少轮结束（end > start）
    parser.add_argument("--model-epoch-end",
                        type=int,
                        default=0,
                        help="The number of epochs the final saved model has been trained.")


    ################################ 数据集与模型命名 ################################
    # 数据集名称
    parser.add_argument("--dataset",
                        type=str,
                        default='simpleqa',
                        help="dataset name: simpleqa, webquestions, grailqa, webqsp, cwq")

    # 模型名（可影响保存路径或加载路径）
    parser.add_argument("--model-name",
                        type=str,
                        default='DiffMatch',
                        help="model name")


    ################################ 数据量控制参数 ################################
    # 每个图生成的合成变体（用于训练）
    parser.add_argument("--num-delta-graphs",
                        type=int,
                        default=100,
                        help="The number of synthetic delta graph pairs for each graph.")

    # 每个图用于测试的图对数量
    parser.add_argument("--num-testing-graphs",
                        type=int,
                        default=100,
                        help="The number of testing graph pairs for each graph.")

    ################################ LLM 设置 ################################
    parser.add_argument("--llm",
                    type=str,
                    default="openchat",
                    help="The LLM to use. Select from: openchat, mistralai7B, nousResearch, mixtral8x, llama2, llama3, gemma, zephyr, starling, qwen, yi")
    
    return parser.parse_args()
