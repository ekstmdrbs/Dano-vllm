import torch
from torch import nn
import torch.nn.functional as F

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
    
    @torch.inference_mode()
    def rejection_sampler(
        self,
        draft_logits: torch.Tensor,
        verification_logits: torch.Tensor
    ) -> list[list[int]]:
        batch_size, num_spec_tokens, vocab_size = draft_logits.shape
        
        # Get probabilities from logits
        draft_probs = F.softmax(draft_logits, dim=-1)
        # print("ASDFASDFASDF")
        # print(verification_logits.shape)
        verification_probs = F.softmax(verification_logits, dim=-1)

        candidate_tokens = torch.argmax(draft_probs, dim=-1)
        # print(candidate_tokens)

        final_accepted_tokens = []

        for i in range(batch_size):
            accepted_tokens_for_seq = []
            all_accepted = True

            for j in range(num_spec_tokens):
                token = candidate_tokens[i, j]
                
                # Get the probability of the candidate token from both models
                p = verification_probs[i, j, token]
                q = draft_probs[i, j, token]

                # Generate a random number for the acceptance check
                r = torch.rand_like(p)

                if r < p / q:
                    # Accept the token
                    accepted_tokens_for_seq.append(token.item())
                else:
                    # Reject this token and all subsequent ones
                    # Resample a single token from the modified distribution (p - q)+
                    diff_probs = torch.clamp(verification_probs[i, j] - draft_probs[i, j], min=0)
                    norm_factor = diff_probs.sum()
                    if norm_factor > 0:
                        resampled_token = torch.multinomial(diff_probs / norm_factor, num_samples=1)
                    else:
                        # If distributions are identical, sample from the target model's distribution
                        resampled_token = torch.multinomial(verification_probs[i, j], num_samples=1)
                    
                    accepted_tokens_for_seq.append(resampled_token.item())
                    all_accepted = False
                    break
            
            # If all candidates were accepted, we sample one more token from the last step
            if all_accepted:
                last_step_logits = verification_logits[i, -1, :]
                final_token = torch.multinomial(F.softmax(last_step_logits, dim=-1), num_samples=1)
                accepted_tokens_for_seq.append(final_token.item())

            final_accepted_tokens.append(accepted_tokens_for_seq)

        return final_accepted_tokens