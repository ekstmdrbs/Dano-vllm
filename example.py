import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

def main():
    #path = os.path.expanduser("/home/ekstmdrbs/models/Qwen3-0.6B")
    path = os.path.expanduser("/home/ekstmdrbs/models/Llama-3.1-8B-Instruct")
    #path = os.path.expanduser("/home/ekstmdrbs/models/EAGLE-LLaMA3.1-Instruct-8B")
    spec_path = os.path.expanduser("/home/ekstmdrbs/models/EAGLE-LLaMA3.1-Instruct-8B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2, speculative_model=spec_path)
    # llm = LLM(path, enforce_eager=True, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0, max_tokens=3)
    
    # Long input
    # with open("input.txt", 'r', encoding='utf-8') as f:
    #     long_line = f.readline()
    #     half_point = len(long_line) // 32
    #     truncated_line = long_line[:half_point]
    # prompts = [truncated_line.strip()]

    # Manual input
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "introduce yourself",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]

    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    
    print(f"\n\n Total Elapsed time: {end_time-start_time}")

if __name__ == "__main__":
    main()
