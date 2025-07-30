from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def commit_accepted_tokens_to_state(self, accepted_ids: list[int]):
        if not accepted_ids:
            return
        self.token_ids.extend(accepted_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def last_token(self) -> int:
        print("Getting last token", self.token_ids[-1])
        return self.token_ids[-1]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        """Returns only the generated token IDs, excluding the prompt."""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        """Returns the number of blocks occupied by cached tokens."""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """Returns the total number of blocks required for all tokens."""
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """Calculates the number of tokens in the last block accurately."""
        num_tokens = len(self.token_ids)
        if num_tokens == 0:
            return 0
        
        result = num_tokens % self.block_size
        return result if result != 0 else self.block_size
    
    def block(self, i: int) -> list[int]:
        """Returns the token IDs for the i-th block."""
        assert 0 <= i < self.num_blocks
        start_idx = i * self.block_size
        return self.token_ids[start_idx : start_idx + self.block_size]