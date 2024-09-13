import random
import re
import random
from typing import List
from data import gsm
from openai import OpenAI
from rich import print
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from data.thinking_styles import thinking_styles
from data.types import Population, EvolutionUnit

multistep_arithmetic_two_examples = gsm.read_jsonl('data/dataset/multistep_arithmetic_two.jsonl')
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda:0",
)
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def zero_order_prompt_gen(unit: EvolutionUnit, problem_description: str, **kwargs) -> EvolutionUnit:
    messages = [{"role": "user", "content": problem_description + " An ordered list of 100 hints: "},]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    result = outputs[0]["generated_text"][-1]

    # search for the pattern "anything after 1. and before 2."
    pattern = r"1\.(.*?)2\."
    match = re.search(pattern, str(result), re.DOTALL)
    if match:
        # return the first match
        unit.P = match.group(1).strip()
    else:
        unit.P = ""

    return unit


def first_order_prompt_gen(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    messages = [{"role": "user",
                 "content": str(unit.M) + " " + str(unit.P)
                 },]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    unit.P = outputs[0]["generated_text"][-1]
    return unit


def lineage_based_mutation(unit: EvolutionUnit, elites: List[EvolutionUnit], **kwargs) -> EvolutionUnit:
    HEADING = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY \n "
    # made a choice not to format it with newlines, could change later.
    ITEMS = "\n".join(["{}. {}".format(i + 1, x.P) for i, x in enumerate(elites)])

    messages = [{"role": "user", "content": HEADING + ITEMS},]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    unit.P = outputs[0]["generated_text"][-1]

    return unit


# Hypermutation
def zero_order_hypermutation(unit: EvolutionUnit, problem_description: str, **kwargs) -> EvolutionUnit:
    RANDOM_THINKING_STYLE = random.sample(thinking_styles, 1)[0]
    messages = [{"role": "user", "content": problem_description + " " + RANDOM_THINKING_STYLE}, ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    unit.M = outputs[0]["generated_text"][-1]
    return unit


def first_order_hypermutation(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    HYPER_MUTATION_PROMPT = "Please summarize and improve the following instruction: "
    messages = [{"role": "user", "content": HYPER_MUTATION_PROMPT + str(unit.M)},]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    unit.M = outputs[0]["generated_text"][-1]

    messages = [{"role": "user", "content": str(unit.M) + " " + str(unit.P)},]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    unit.P = outputs[0]["generated_text"][-1]
    return unit


# Lamarckian Mutation
def working_out_task_prompt(unit: EvolutionUnit, **kwargs) -> EvolutionUnit:
    RANDOM_WORKING_OUT = random.sample(multistep_arithmetic_two_examples, 1)[0]
    messages = [{
        "role": "user",
        "content": "I gave a friend an instruction and some advice. Here are the correct examples of his workings out "
                   + RANDOM_WORKING_OUT['question'] + " " + RANDOM_WORKING_OUT['answer'] + " The instruction was: "
    }, ]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    unit.P = outputs[0]["generated_text"][-1]
    return unit


# omitting the estimation_distribution_mutation
MUTATORS = [
    zero_order_prompt_gen,
    first_order_prompt_gen,
    lineage_based_mutation,
    zero_order_hypermutation,
    first_order_hypermutation,
    working_out_task_prompt
]


def mutate(population: Population) -> Population:

    indices = [i for i in range(len(population.units))]
    random.shuffle(indices)
    pairs = [indices[2 * x:2 * x + 2] for x in range(len(indices) // 2)]

    # binary tourmanent genetic algorithm
    for i in range(len(pairs)):

        first_unit = population.units[pairs[i][0]]
        second_unit = population.units[pairs[i][1]]
        FIRST_WON = False
        if first_unit.fitness >= second_unit.fitness:
            # loser gets mutated.
            FIRST_WON = True
            mutation_input = second_unit
        else:
            mutation_input = first_unit

        data = {
            'unit': mutation_input,
            'elites': population.elites,
            'problem_description': population.problem_description,
        }

        # uniformly pick and call a random mutation operator on the losing unit
        random_mutator = random.sample(MUTATORS, 1)[0]
        print(f"MUTATING: {mutation_input} with {random_mutator.__name__}")

        random_mutator(**data)

    return population