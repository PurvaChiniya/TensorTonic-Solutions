import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Returns: None
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        scale = d_model ** -0.5

        self.Wq = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.Wk = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.Wv = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.Wo = nn.Parameter(torch.randn(d_model, d_model) * scale)

    def forward(self, Q, K, V):
        """
        Returns: output tensor
        
        """
        batch_size = K.shape[0]
         
        
        q = Q @ self.Wq
        k = K @ self.Wk
        v = V @ self.Wv
        # (B, L, d_model) -> (B, H, L, d_k)
        q = q.reshape(
            batch_size, Q.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)
        k = k.reshape(
            batch_size, K.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)

        v = v.reshape(
            batch_size, V.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)


        scores = q@k.transpose(-1, -2)
        scores = scores/ (self.d_k**0.5)
        attention_softmax = torch.softmax(scores, dim = -1)
        head_outputs = attention_softmax@v
        concatenated = head_outputs.transpose(1, 2).contiguous()
        concatenated = concatenated.reshape(
            batch_size, Q.shape[1], self.d_model
        )

        # this is for a single head now we need to project to outptu 
        return concatenated@self.Wo