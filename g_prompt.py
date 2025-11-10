import torch
import torch.nn as nn

class GPrompt(nn.Module):
    def __init__(self, num_heads, embed_dim, prompt_length, prompt_init='xavier'):
        """
        General Prompt

        Args:
            num_heads: Number of heads in the multi-head attention
            embed_dim: Embedding dimension
            prompt_length: Length of the prompt
            prompt_init: Initialization of the prompt
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.head_dim = embed_dim // num_heads
    
        self.prompt_k = nn.Parameter(torch.empty(1, num_heads, prompt_length, self.head_dim))
        self.prompt_v = nn.Parameter(torch.empty(1, num_heads, prompt_length, self.head_dim))
        
        self.initialise(prompt_init)

    def initialise(self, method):
        """
        Initialize the prompt

        Args:
            method: Initialization method
        """
        if method == 'uniform':
            nn.init.uniform_(self.prompt_k)
            nn.init.uniform_(self.prompt_v)
        elif method == 'normal':
            nn.init.normal_(self.prompt_k)
            nn.init.normal_(self.prompt_v)
        elif method == 'xavier':
            nn.init.xavier_uniform_(self.prompt_k)
            nn.init.xavier_uniform_(self.prompt_v)
        elif method == 'zeros':
            nn.init.zeros_(self.prompt_k)
            nn.init.zeros_(self.prompt_v)
        else:
            raise ValueError('Invalid initialization method, default didn\'t trigger')
        

    def get_prompt(self):
        """
        Get the prompt

        Returns:
            Tuple: Prompt key and prompt value
        """
        return self.prompt_k, self.prompt_v