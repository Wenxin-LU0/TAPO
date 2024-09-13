import os
import time
from datetime import datetime
import csv
import torch
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rich import print
import numpy as np

from data.types import Population

# 引入所需库
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from typing import Tuple
import nltk

# 初始化模型
model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large')


def get_embedding(text: str) -> list:
    """获取文本的嵌入向量"""
    return model.encode(text)


def calculate_similarity(embedding1: list, embedding2: list) -> float:
    """计算两个嵌入向量之间的余弦相似度"""
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity


def calculate_complexity(text: str) -> Tuple[int, int, int]:
    """计算文本的复杂性，包括长度、语法复杂性和逻辑步骤"""
    words = text.split()
    length = len(words)

    # 计算语法复杂性（例如从句数量）
    sentences = nltk.sent_tokenize(text)
    grammar_complexity = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)

    # 逻辑步骤（可以根据特定任务定义）
    logic_steps = text.count('.')

    return length, grammar_complexity, logic_steps


def calculate_perplexity(text: str) -> float:
    """计算文本的困惑度的指标"""
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    tokenize_input = gpt2_tokenizer.encode(text, return_tensors="pt")
    loss = gpt2_model(tokenize_input, labels=tokenize_input).loss
    return torch.exp(loss).item()


def _evaluate_fitness(population: Population, results, batch, num_evals):
  start_time = time.time()

  now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  filename = f"{now}.csv"
  result_dir = population.result_dir
  filepath = os.path.join(result_dir, filename)

  with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['unit_index', 'question', 'generated_answer', 
            'correct_answer', 'prompt', 'total_fitness',
            'similarity', 'complexity','log_perplexity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # 初始化fitness
    for unit in population.units:
      unit.fitness = 0
    # 开始评分
    for unit_index, fitness_results in enumerate(results):
      for i, x in enumerate(fitness_results):
        generated_answer = x
        print(f"正在评价结果：{i}")
        correct_answer = batch[i % num_evals]['answer']  # 序号取余才是真正的问题标号
        question = batch[i % num_evals]['question']

        # similarity evaluation
        similarity = calculate_similarity(get_embedding(generated_answer), get_embedding(correct_answer))
        # Complexity evaluation
        length, grammar_complexity, logic_steps = calculate_complexity(generated_answer)
        # Perplexity evaluation
        perplexity = calculate_perplexity(generated_answer)
        log_perplexity = np.log(perplexity + 0.1)

        # 综合评价
        weights = {
          'similarity_weight': 1,
          'complexity_weight': -0.1,
          'log_perplexity_weight': -0.1
        }
        unit.fitness += (
          weights['similarity_weight'] * similarity +
          weights['complexity_weight'] * (length + grammar_complexity + logic_steps)
          + weights['log_perplexity_weight'] * log_perplexity
        )

        # 记录日志和结果
        writer.writerow({
          'unit_index': unit_index,
          'question': question,
          'generated_answer': generated_answer,
          'correct_answer': correct_answer,
          'prompt': unit.P,
          'total_fitness': unit.fitness,
          'similarity': similarity,
          'log_perplexity': log_perplexity,
          'complexity': length + grammar_complexity + logic_steps
        })
      # 记录种群的最佳效果
      elite_fitness = max(unit.fitness for unit in population.units)
      population.elites.extend([unit for unit in population.units if unit.fitness == elite_fitness])

  end_time = time.time()

  return population
