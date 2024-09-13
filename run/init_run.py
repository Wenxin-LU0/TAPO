import os
from datetime import datetime
from typing import List
import transformers
import torch
from openai import OpenAI

from data import gsm
from data.types import EvolutionUnit, Population


def hello():
  print("hello!")

def create_population(tp_set: List, mutator_set: List, problem_description: str):

    data = {
        'size': len(tp_set)*len(mutator_set),
        'age': 0,
        'problem_description': problem_description,
        'elites': [],
        'units': [EvolutionUnit(**{
            'T': t,
            'M': m,
            'P': '',
            'fitness': 0,
            'history': []
        }) for t in tp_set for m in mutator_set],
        'result_dir': ''
    }
    return Population(**data)

def init_run(population:Population,model:OpenAI,batch):
   # 初始化当前时间目录以存储结果
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "Output")
    current_time_dir = os.path.join(output_dir, now)
    if not os.path.exists(current_time_dir):
        os.makedirs(current_time_dir)
    population.result_dir = current_time_dir

    # 生成初始prompt
    print("开始生成初始prompt")
    prompts = []
    for unit in population.units:
        template = f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        completion = model.chat.completions.create(
            model="meta/llama3-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": template,
                },
            ],
        )
        prompts.append(completion.choices[0].message.content)

    # 将生成的prompt存储到units.P中
    assert len(prompts) == population.size, "size of google response to population is mismatched"
    for i, item in enumerate(prompts):
        population.units[i].P = item
    print("改好了")
    print(prompts)
    return prompts

    
