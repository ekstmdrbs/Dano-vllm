# Install libraries
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git

# How to use
In the root directory execute example.py

''' python
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2, speculative_model=spec_path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0, max_tokens=3)
'''
Use either one of two lines
1. vanilla decoding -> Second line
   With config.py -> gpu_memory_utilization: float = 0.9
2. speculative decoding -> First line
   With config.py -> gpu_memory_utilization: float = 0.6

# Notification
Some codes are hard coded e.g. model initiation fixed with Llama3.1 as target and EAGLE-Llama3.1
The model's weights are loaded with the actual weight file saved. Need to modify the path for personal usage
c.f. Eagle model's tokenizer file copied from Llama3.1 and it additionally loads lm_head fully on rank0 from Llama's weight during runtime
