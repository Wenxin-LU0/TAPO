from openai import OpenAI
import argparse
import random
from rich import print

from pb import create_population, init_run, run_for_n, gsm
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles


dataset_examples = gsm.read_jsonl('pb/data/gsm.jsonl')

parser = argparse.ArgumentParser(description='Run the TAPO Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', type=int, default=2)
parser.add_argument('-ts', '--num_thinking_styles', type=int, default=4)
parser.add_argument('-e', '--num_evals', type=int, default=1)
parser.add_argument('-n', '--simulations', type=int, default=3)
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")

args = vars(parser.parse_args())
total_evaluations = args['num_mutation_prompts'] * args['num_thinking_styles'] * args['num_evals']
tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
mutator_set = thinking_styles[:int(args['num_thinking_styles'])]
batch = dataset_examples[:int(args['num_evals'])]
question_set = random.sample(dataset_examples,int(args['num_evals']))
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

client = OpenAI(api_key='sk-proj-xlWuNWvRP2NZ3NFlYzPTKyCjfdj87s1q5IL1HAL_V'
                    '-oxmMyIkwB_R6MpWb09f9j_BvdWlNPld8T3BlbkFJiMPKQq2dTogQ7ejOoTdX34BLM-0OFN3HmxKvr7c4onMaGYg_x'
                    '-tOr9nRjvcuUamQx28WmbMRcA')
model_id = "gpt-4o-2024-08-06"

init_run(p, client, model_id, batch, int(args['num_evals']))
run_for_n(int(args['simulations']), p, client, model_id, batch, int(args['num_evals']))

print("done processing! final gen:")
print(p.units)
