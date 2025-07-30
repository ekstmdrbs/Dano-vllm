import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model, load_eagle_model
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.llama3_1 import LlamaForCausalLM
from nanovllm.models.llama3_1_eagle import EagleLlama3_1ForCausalLM
import numpy as np


class ModelRunner:
    def __init__(self, config: Config, model_cls: type[torch.nn.Module], rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        print(f"[DEBUG] Llama (Target) model config dtype: {hf_config.torch_dtype}")

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.last_hidden_stage = None
        self.spec_step = config.num_speculative_tokens
        
        if self.world_size > 1:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

            tp_size = dist.get_world_size()
            all_ranks_for_tp = list(range(tp_size))
            tp_group = dist.new_group(ranks=all_ranks_for_tp)

            hf_config.tp_group = tp_group
            hf_config.tp_rank = dist.get_rank()
            hf_config.tp_size = dist.get_world_size()
        
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        self.model = model_cls(hf_config)
        # print(f"[Rank {self.rank}] Loading Target Model Weights...")
        load_model(self.model, config.model)
        print(f"[DEBUG] Llama's actual weight dtype: {self.model.model.embed_tokens.weight.dtype}")

        # print(f"[Rank {self.rank}] Target Model Loaded.")
        self.sampler = Sampler()

        partial_weights = self.model.lm_head.weight

        if self.world_size > 1:
            all_weights_list = [torch.empty_like(partial_weights) for _ in range(self.world_size)]
            dist.all_gather(all_weights_list, partial_weights, group=hf_config.tp_group)
        else:
            all_weights_list = [partial_weights]

        self.draft_model = None

        if self.rank == 0 and config.speculative_model:
            # print(f"[Rank 0] Setting up Draft Model...")
            from transformers import AutoConfig
            import torch.nn as nn

            draft_model_name = config.speculative_model
            draft_hf_config = AutoConfig.from_pretrained(draft_model_name)
            TARGET_DTYPE = config.hf_config.torch_dtype
            draft_hf_config.torch_dtype = TARGET_DTYPE
            print(f"[DEBUG] Eagle (Draft) model config dtype: {draft_hf_config.torch_dtype}")
            draft_hf_config.tie_word_embeddings = False
            self.draft_model = EagleLlama3_1ForCausalLM(draft_hf_config)
            load_eagle_model(self.draft_model, draft_model_name)
            
            self.draft_model.lm_head = nn.Linear(
                draft_hf_config.hidden_size, hf_config.vocab_size, bias=False
            ).to(device="cuda", dtype=hf_config.torch_dtype)

            full_weights = torch.cat(all_weights_list, dim=0)
            
            self.draft_model.lm_head.weight.data.copy_(full_weights)
            print(f"[DEBUG] Eagle's actual weight dtype: {self.draft_model.model.embed_tokens.weight.dtype}")
            # print(f"[Rank 0] Draft model setup complete.")

        if self.world_size > 1:
            dist.barrier()

        self.warmup_model()
        # print(f"[Rank {self.rank}] Warmed up. Let's fucking go")
        dist.barrier()
        self.allocate_kv_cache()

        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        # print(f"[Rank {self.rank}]Just captured cuda graph")

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            # print(f"[Rank {self.rank}]Wait a sec I read something", method_name)
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()
    
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        print(f"[Rank {self.rank}]Prefill input_ids: {len(input_ids)}, positions: {len(positions)}, slot_mapping: {len(slot_mapping)}")
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        print("AAAA",cu_seqlens_q,cu_seqlens_k,input_ids, positions,flush=True)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens)
        print(f"[Rank {self.rank}]Preparing decode with input_ids: {input_ids}, positions: {positions}, slot_mapping: {slot_mapping}, context_lens: {context_lens}")
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, True, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        print(input_ids,positions)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def prepare_verify(self, seqs: list[Sequence], candidate_tokens: torch.Tensor):
        # print(f"[Rank {self.rank}]Entered prepare_verify")
        batch_size, num_candidates = candidate_tokens.shape
        
        input_ids_list = []
        positions_list = []
        slot_mapping = []

        for i, seq in enumerate(seqs):
            seq_candidate_tokens = candidate_tokens[i]
            input_ids_list.extend(seq_candidate_tokens.tolist())
            
            start_pos = len(seq)
            positions_list.extend(list(range(start_pos, start_pos + num_candidates)))

            current_block_idx = seq.block_table[-1]
            token_offset_in_block = seq.last_block_num_tokens
            
            for j in range(num_candidates):
                if token_offset_in_block >= self.block_size:
                    current_block_idx = seq.block_table[seq.num_blocks - 1 + (j // self.block_size)]
                    token_offset_in_block = 0

                slot = current_block_idx * self.block_size + token_offset_in_block
                slot_mapping.append(slot)
                token_offset_in_block += 1

        # input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
        # positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
        # slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")
        # print(f"[Rank {self.rank}]Preparing verify with input_ids: {input_ids.shape}, positions: {positions.shape}, slot_mapping: {slot_mapping_tensor.shape}")
        # print(f"[Rank {self.rank}]Slot mapping: {slot_mapping_tensor.tolist()} for input_ids: {input_ids.tolist()} and positions: {positions.tolist()}")
        # block_tables = self.prepare_block_tables(seqs)

        # max_seqlen_q = num_candidates

        # cu_seqlens_q = torch.arange(
        #     0, (batch_size + 1) * num_candidates, 
        #     step=num_candidates, dtype=torch.int32, device="cuda"
        # )
        # set_context(
        #     is_prefill=True,
        #     max_seqlen_q=max_seqlen_q,
        #     max_seqlen_k=max_seqlen_q,
        #     cu_seqlens_q=cu_seqlens_q,
        #     cu_seqlens_k=cu_seqlens_q,
        #     slot_mapping=slot_mapping_tensor,
        #     block_tables=block_tables,
        #     write_cache=False
        # )
        cu_seqlens_q_list = [0]
        cu_seqlens_k_list = [0]
        max_seqlen_q = num_candidates
        max_seqlen_k = 0

        for seq in seqs:
            # Query의 길이는 항상 후보 토큰의 개수입니다.
            seqlen_q = num_candidates
            cu_seqlens_q_list.append(cu_seqlens_q_list[-1] + seqlen_q)

            # Key의 길이는 기존 토큰 + 후보 토큰의 개수, 즉 전체 길이입니다.
            seqlen_k = len(seq.token_ids) + num_candidates
            cu_seqlens_k_list.append(cu_seqlens_k_list[-1] + seqlen_k)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
        
        print(f"[Rank {self.rank}] Preparing Verify Slot mapping: {slot_mapping} for input_ids: {input_ids_list} and positions: {positions_list}")
        # 텐서로 변환
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device="cuda")
        block_tables = self.prepare_block_tables(seqs)
        
        # Context에 올바른 K/V 길이를 전달
        set_context(
            is_prefill=True,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,   # ⬅️ 올바른 K 최대 길이
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,   # ⬅️ 올바른 K 누적 길이
            slot_mapping=slot_mapping_tensor,
            block_tables=block_tables,
            write_cache=False
        )
        print("ASDF", input_ids ,cu_seqlens_q,cu_seqlens_k, slot_mapping_tensor, positions)
        # print(f"[Rank {self.rank}]End of prepare_verify")
        return input_ids, positions
    
    def prepare_commit(self, seqs: list[Sequence], accepted_tokens_list: list[list[int]]):
        input_ids_list, positions_list, slot_mapping = [], [], []
        cu_seqlens_q = [0]
        max_seqlen_q = 0

        if not any(accepted_tokens_list):
            return None, None

        for i, seq in enumerate(seqs):
            tokens_to_commit = accepted_tokens_list[i]
            num_accepted = len(tokens_to_commit)
            if num_accepted == 0:
                cu_seqlens_q.append(cu_seqlens_q[-1])
                continue

            input_ids_list.extend(tokens_to_commit)
            start_pos = len(seq.token_ids)
            positions_list.extend(range(start_pos, start_pos + num_accepted))

            token_offset_in_block = seq.last_block_num_tokens
            current_block_table_idx = (len(seq.token_ids) -1) // self.block_size if len(seq.token_ids) > 0 else 0

            for j in range(num_accepted):
                if token_offset_in_block >= self.block_size:
                    token_offset_in_block = 0
                    current_block_table_idx += 1
                
                block_idx = seq.block_table[current_block_table_idx]
                slot = block_idx * self.block_size + token_offset_in_block
                slot_mapping.append(slot)
                token_offset_in_block += 1
            
            cu_seqlens_q.append(cu_seqlens_q[-1] + num_accepted)
            max_seqlen_q = max(max_seqlen_q, num_accepted)

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
        positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
        block_tables = self.prepare_block_tables(seqs)
        print(f"[Rank {self.rank}]Preparing commit with input_ids: {input_ids.shape}, positions: {positions.shape}, slot_mapping: {slot_mapping_tensor.shape}, cu_seqlens_q: {cu_seqlens_q.shape}")
        print(f"[Rank {self.rank}]Slot mapping: {slot_mapping_tensor.tolist()} for input_ids: {input_ids.tolist()} and positions: {positions.tolist()}")
        set_context(
            is_prefill=True,
            write_cache=True,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            slot_mapping=slot_mapping_tensor,
            block_tables=block_tables
        )
        
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, is_verify: bool = False):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # print(f"[Rank {self.rank}]prefill model_output enter")
            print("The context is:",get_context())
            model_output = self.model(input_ids, positions)
            print(f"[Rank {self.rank}]Model output: {model_output}")
            # print(f"[Rank {self.rank}]prefill mmodel.compute_logits enter")
            logits = self.model.compute_logits(model_output,is_verify)
            return logits, model_output
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()

            model_output = graph_vars["outputs"][:bs]
            logits = self.model.compute_logits(model_output,is_verify)
            return logits, model_output


    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:
        # --- NON-SPECULATIVE PATH ---
        if self.config.speculative_model is None:
            get_context().write_cache = True
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            print(f"[Rank {self.rank}]Input IDs: {input_ids}, Positions: {positions}")
            if not is_prefill:
                has_nan_in_cache = torch.isnan(self.kv_cache).any()
                print(f"\n[DEBUG] Before Prefill, is there NaN in KV cache? -> {has_nan_in_cache}\n")
            logits, checker = self.run_model(input_ids, positions, is_prefill, is_verify = False)
            print(f"Non-speculative's hidden state : {checker}")

            accepted_token_ids = []
            if self.rank == 0:
                temperatures = self.prepare_sample(seqs)
                accepted_token_ids = [[t] for t in self.sampler(logits, temperatures).tolist()]
            
            if self.world_size > 1:
                if self.rank == 0:
                    data = pickle.dumps(accepted_token_ids)
                    size_tensor = torch.tensor([len(data)], dtype=torch.long, device="cuda")
                else:
                    size_tensor = torch.empty(1, dtype=torch.long, device="cuda")

                dist.broadcast(size_tensor, src=0)

                if self.rank != 0:
                    data_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cuda")
                else:
                    data_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).cuda()
                
                dist.broadcast(data_tensor, src=0)

                if self.rank != 0:
                    accepted_token_ids = pickle.loads(data_tensor.cpu().numpy().tobytes())

            reset_context()
            return accepted_token_ids

        # --- SPECULATIVE DECODING PATH ---
        # Prefill Step
        if is_prefill:
            get_context.write_cache = True
            # print(f"[Rank {self.rank}] Entered is_prefill path", flush=True)
            input_ids, positions = self.prepare_prefill(seqs)
            logits, model_output = self.run_model(input_ids, positions, is_prefill=True, is_verify=False)
            print(f"speculative's hidden state : {model_output}")
            context = get_context()
            last_token_indices = context.cu_seqlens_q[1:] - 1
            self.last_hidden_state = model_output[last_token_indices]

            accepted_token_ids = []
            if self.rank == 0:
                temperatures = self.prepare_sample(seqs)
                accepted_token_ids = [[t] for t in self.sampler(logits, temperatures).tolist()]

            if self.world_size > 1:
                if self.rank == 0:
                    data = pickle.dumps(accepted_token_ids)
                    size_tensor = torch.tensor([len(data)], dtype=torch.long, device="cuda")
                else:
                    size_tensor = torch.empty(1, dtype=torch.long, device="cuda")
                dist.broadcast(size_tensor, src=0)
                if self.rank == 0:
                    data_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).cuda()
                else:
                    data_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cuda")
                dist.broadcast(data_tensor, src=0)
                if self.rank != 0:
                    accepted_token_ids = pickle.loads(data_tensor.cpu().numpy().tobytes())

            # print(f"[Rank {self.rank}] Finished is_prefill path", flush=True)
            reset_context()
            return accepted_token_ids
        
        # Decode Step
        else:
            # print(f"[Rank {self.rank}] Entered Decode path")
            batch_size = len(seqs)
            if self.last_hidden_state is not None and self.last_hidden_state.shape[0] != batch_size:
                print(f"\n\n\n\nWarning: Trimming hidden_state from batch size {self.last_hidden_state.shape[0]} to {batch_size}\n\n\n")
                self.last_hidden_state = self.last_hidden_state[:batch_size]

            input_ids, positions = self.prepare_decode(seqs)
            print(f"[Rank {self.rank}]Input IDs: {input_ids}, Positions: {positions}")
            # Step 1: Propose on Rank 0
            if self.rank == 0:
                last_tokens = [s.last_token for s in seqs]
    
                # Get the starting positions for the new tokens
                # This assumes 'positions' was prepared by prepare_decode
                start_positions = positions 
                # print(f"[Rank {self.rank}]Start Proposing in Rank 0")
                print(f"[Rank {self.rank}]Hidden state before proposing: {self.last_hidden_state}")
                candidate_tokens, draft_logits = self.draft_model.propose(
                    last_tokens,
                    self.last_hidden_state, # Pass the state from the previous step
                    self.spec_step,
                    start_positions
                )
                # print(f"[Rank {self.rank}]End of Proposing in Rank 0")
            else:
                # Create empty tensors on other ranks to receive the broadcasted data
                # print(f"[Rank {self.rank}]Rank1 is getting ready for broadcast", flush=True)
                candidate_tokens = torch.empty((batch_size, self.spec_step), dtype=torch.long, device="cuda")
                draft_logits = torch.empty((batch_size, self.spec_step, self.config.hf_config.vocab_size), dtype=self.config.hf_config.torch_dtype, device="cuda")

            # Step 2: Share Candidates
            # print(f"[Rank {self.rank}]Step2")
            if self.world_size > 1:
                # print(f"[Rank {self.rank}]Broadcasting candidate tokens and draft logits")
                dist.barrier()  # Ensure all ranks are ready before broadcasting
                dist.broadcast(candidate_tokens, src=0)
                dist.broadcast(draft_logits, src=0)
                print(f"[Rank {self.rank}] The candidate tokens and draft_logits value is {candidate_tokens} and {draft_logits}", flush=True)

            # Step 3: Verify on All Ranks
            # print(f"[Rank {self.rank}]Step3")
            input_ids, positions = self.prepare_verify(seqs, candidate_tokens)
            print(f"[Rank {self.rank}]Input IDs for verification: {input_ids}, Positions for verification: {positions}")

            has_nan_in_cache = torch.isnan(self.kv_cache).any()
            print(f"\n[DEBUG] Before Prefill, is there NaN in KV cache? -> {has_nan_in_cache}\n")
            
            verification_logits, verification_output = self.run_model(input_ids, positions, is_prefill=True, is_verify=True)
            # print("Verification logits shape:", verification_logits.shape)
            print(f"speculative's hidden state : {verification_output}")

            # Step 4: Decide on Rank 0
            # print(f"[Rank {self.rank}]Step4")
            accepted_token_ids = []
            if self.rank == 0:
                batch_size = len(seqs)
                num_spec_tokens = self.spec_step 
                verification_logits = verification_logits.view(batch_size, num_spec_tokens, -1)

                accepted_token_ids = self.sampler.rejection_sampler(draft_logits, verification_logits)
                accepted_counts = torch.tensor([len(tokens) for tokens in accepted_token_ids], dtype=torch.long, device="cuda")
            else:
                accepted_counts = torch.empty(batch_size, dtype=torch.long, device="cuda")
            
            # Step 5: Broadcast the decision (number of accepted tokens)
            # print(f"[Rank {self.rank}]Step5")
            if self.world_size > 1:
                dist.broadcast(accepted_counts, src=0)
            
            # Step 6: Commit to Cache & Finalize State
            tokens_to_commit_list = []
            if self.rank == 0:
                tokens_to_commit_list = accepted_token_ids
            else:
                for i in range(batch_size):
                    num_accepted = accepted_counts[i].item()
                    tokens_to_commit_list.append(candidate_tokens[i, :num_accepted].tolist())
            
            commit_input_ids, commit_positions = self.prepare_commit(seqs, tokens_to_commit_list)
            
            if commit_input_ids is not None:
                with torch.no_grad():
                    self.run_model(commit_input_ids, commit_positions, is_prefill=True, is_verify=False)
            
            # for i, seq in enumerate(seqs):
            #     seq.commit_accepted_tokens_to_state(tokens_to_commit_list[i])

            verification_output_reshaped = verification_output.view(batch_size, self.spec_step, -1)
            batch_indices = torch.arange(batch_size, device="cuda")
            last_accepted_indices = accepted_counts - 1


            last_accepted_indices.clamp_(min=0)
            self.last_hidden_state = verification_output_reshaped[batch_indices, last_accepted_indices]
            
            # if self.rank == 0:
            #     print(f"[Rank {self.rank}]Finished Decode path with accepted tokens: {accepted_token_ids}")

            # if self.rank != 0:
            #         data_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cuda")
            # else:
            #     data_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).cuda()
                
            #     dist.broadcast(data_tensor, src=0)

            # if self.rank != 0:
            #     accepted_token_ids = pickle.loads(data_tensor.cpu().numpy().tobytes())

            reset_context()
            return accepted_token_ids if self.rank == 0 else []


    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
