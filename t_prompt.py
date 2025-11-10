import torch
import torch.nn as nn

class TPrompt(nn.Module):
    def __init__(self, num_heads, embed_dim, prompt_init='xavier'):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.prompts_k = nn.ParameterList([
            nn.Parameter(torch.empty(1,num_heads, 1, self.head_dim))
            for _ in range(2)#there are 2 prompts, nextActivity and finalOutcome
        ])
        self.prompts_v = nn.ParameterList([
            nn.Parameter(torch.empty(1,num_heads, 1, self.head_dim))
            for _ in range(2)#nextActivity and finalOutcome
        ])
        self.init_prompts(prompt_init)

    def init_prompts(self, method):
        for k,v in zip(self.prompts_k, self.prompts_v):
            if method == 'uniform':
                nn.init.uniform_(k)
                nn.init.uniform_(v)
            elif method == 'normal':
                nn.init.normal_(k)
                nn.init.normal_(v)
            elif method == 'xavier':
                nn.init.xavier_uniform_(k)
                nn.init.xavier_uniform_(v)
            elif method == 'zeros':
                nn.init.zeros_(k)
                nn.init.zeros_(v)

    def get_prompt(self, prompt_idx):
        return self.prompts_k[prompt_idx], self.prompts_v[prompt_idx]
