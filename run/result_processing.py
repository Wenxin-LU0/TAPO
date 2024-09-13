from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def handle_result(MAX_handling, system_prompt, batch_example):
  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = "left")
  tokenizer.pad_token_id = tokenizer.eos_token_id
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")

  tokenizer.pad_token_id = tokenizer.eos_token_id
  terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  tasks_len = 0
  total_tasks_list = []
  result_list = []

  # 创建整体任务集
  for count, prompt in enumerate(system_prompt):
      for index, example in enumerate(batch_example):
          description = example['question']
          input_task = [
              [
                  {"role": "system", "content": prompt},
                  {"role": "user", "content": description},
              ],
          ]
          total_tasks_list.append(input_task)
          tasks_len += 1
  print("批处理文件创建完成,即将进行结果生成")
  print(total_tasks_list)

  # 进行batch拆分
  i_pre = 0
  i_now = MAX_handling
  j = 1  # 为循环计数，并为分割文件编号

  while i_pre <= tasks_len:
      if i_now >= tasks_len: i_now = tasks_len
      # 进行批处理
      for myinput in total_tasks_list[i_pre:i_now]:
          texts = tokenizer.apply_chat_template(myinput, add_generation_prompt=True, tokenize=False)
          inputs = tokenizer(texts, padding="longest", return_tensors="pt")
          inputs = {key: val.cuda() for key, val in inputs.items()}
          temp_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

          gen_tokens = model.generate(
              **inputs,
              max_new_tokens=512,
              pad_token_id=tokenizer.eos_token_id,
              eos_token_id=terminators,
              do_sample=True,
              temperature=0.6,
              top_p=0.9
          )

          gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
          gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]
          result_list.append(gen_text)
          
          print(gen_text)

      i_pre += MAX_handling
      i_now += MAX_handling
      j += 1
  print("批处理结束")
  return result_list
