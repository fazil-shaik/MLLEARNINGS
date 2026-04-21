import numpy as np
from typing import Tuple

class SimpleTransformer:
    """Minimal transformer for text sequence-to-sequence tasks"""
    
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, d_ff: int = 256, max_seq_len: int = 100):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Embedding layer: vocab_size → d_model dimensions
        self.embed = np.random.randn(vocab_size, d_model) * 0.01
        
        # Positional encoding: hardcoded sinusoidal (no learning)
        self.pos_enc = self._positional_encoding(max_seq_len, d_model)
        
        # Transformer parameters (simplified: 1 attention head per layer for clarity)
        self.W_q = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        self.W_k = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        self.W_v = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        self.W_o = [np.random.randn(d_model, d_model) * 0.01 for _ in range(num_layers)]
        
        # Feed-forward networks
        self.W_ff1 = [np.random.randn(d_model, d_ff) * 0.01 for _ in range(num_layers)]
        self.W_ff2 = [np.random.randn(d_ff, d_model) * 0.01 for _ in range(num_layers)]
        self.b_ff1 = [np.zeros(d_ff) for _ in range(num_layers)]
        self.b_ff2 = [np.zeros(d_model) for _ in range(num_layers)]
    
    def _positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Generate sinusoidal positional encoding"""
        pos = np.arange(seq_len)[:, np.newaxis]
        dim = np.arange(d_model)[np.newaxis, :]
        angle = pos / np.power(10000, (2 * (dim // 2)) / d_model)
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angle[:, 0::2])  # even dims
        pe[:, 1::2] = np.cos(angle[:, 1::2])  # odd dims
        return pe
    
    def _attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled dot-product attention"""
        # Q, K, V shape: (seq_len, d_model)
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)  # (seq_len, seq_len)
        
        # Causal mask: prevent looking at future tokens (in decoder-only)
        mask = np.tril(np.ones_like(scores))
        scores = np.where(mask, scores, -1e9)
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Apply to values
        output = np.dot(attn_weights, V)  # (seq_len, d_model)
        return output, attn_weights
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Forward pass through the transformer
        
        Args:
            token_ids: (seq_len,) integer token indices
        
        Returns:
            logits: (seq_len, vocab_size) unnormalized predictions
        """
        seq_len = len(token_ids)
        
        # 1. Embedding + positional encoding
        x = self.embed[token_ids] + self.pos_enc[:seq_len]  # (seq_len, d_model)
        
        # 2. Transformer layers
        for layer in range(self.num_layers):
            # Multi-head attention (simplified: 1 head)
            Q = np.dot(x, self.W_q[layer])
            K = np.dot(x, self.W_k[layer])
            V = np.dot(x, self.W_v[layer])
            
            attn_out, _ = self._attention(Q, K, V)
            attn_out = np.dot(attn_out, self.W_o[layer])
            
            # Residual + layer norm (approximate)
            x = x + attn_out
            x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6)
            
            # Feed-forward
            ff_out = np.dot(x, self.W_ff1[layer]) + self.b_ff1[layer]
            ff_out = np.maximum(ff_out, 0)  # ReLU
            ff_out = np.dot(ff_out, self.W_ff2[layer]) + self.b_ff2[layer]
            
            # Residual + layer norm
            x = x + ff_out
            x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6)
        
        # 3. Output projection
        logits = np.dot(x, self.embed.T)  # (seq_len, vocab_size)
        return logits
    
    def decode(self, token_ids: np.ndarray, max_new_tokens: int = 10) -> list:
        """Greedy generation: extend sequence one token at a time"""
        generated = list(token_ids)
        
        for _ in range(max_new_tokens):
            logits = self.forward(np.array(generated))
            next_token = np.argmax(logits[-1])  # greedy: pick highest prob
            generated.append(next_token)
        
        return generated


# Example usage
if __name__ == "__main__":
    # Toy vocabulary
    vocab = {0: "<pad>", 1: "hello", 2: "world", 3: "goodbye", 4: "earth", 5: "!"}
    
    # Create transformer
    model = SimpleTransformer(vocab_size=6, d_model=32, num_layers=2, num_heads=1)
    
    # Input: "hello world"
    input_ids = np.array([1, 2])
    
    # Forward pass
    logits = model.forward(input_ids)
    print(f"Input tokens: {[vocab[i] for i in input_ids]}")
    print(f"Logits shape: {logits.shape}")
    print(f"Next token prediction: {vocab[np.argmax(logits[-1])]}")
    
    # Generate continuation
    output_ids = model.decode(input_ids, max_new_tokens=3)
    print(f"Generated sequence: {[vocab[i] for i in output_ids]}")