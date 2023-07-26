#![allow(dead_code)]

/// The inference (or is it model?) configuration
struct Config {
    /// Transformer Dimension
    dim: u32,
    /// Number of hidden dimensions for the Feed Forward Network
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    seq_len: u32,
}

// TODO(alvaro): Can we make the size of the token embedding table a const generic?
// TODO(alvaro): Can we make the float generic to allow for simple quantization?

/// The parsed weights for the transformer
struct TransformerWeights {
    /// Token Embedding Table (vocab_size, dim)
    token_embedding_table: Vec<f32>,
    /// Attention RMSNorm weights (layer, dim)
    rms_att_weight: Vec<f32>,
    /// FFN RMSNorm weights (layer, dim)
    rms_ffn_weight: Vec<f32>,
    /// Weights for Q matmul (layer, dim, dim)
    wq: Vec<f32>,
    /// Weights for K matmul (layer, dim, dim)
    wk: Vec<f32>,
    /// Weights for V matmul (layer, dim, dim)
    wv: Vec<f32>,
    /// Weights for O matmul (layer, dim, dim)
    wo: Vec<f32>,
    /// Weights for FFN layer 1 (layer, hidden_dim, dim)
    w1: Vec<f32>,
    /// Weights for FFN layer 3 (layer, dim, hidden_dim)
    w2: Vec<f32>,
    /// Weights for FFN layer 2 (layer, hidden_dim, dim)
    w3: Vec<f32>,
    /// Weights for final RMSNorm (dim,)
    rms_final_weight: Vec<f32>,
    /// Real part for freq_cis for RoPE relatively positional embeddings (seq_len, dim/2)
    freq_cis_real: Vec<f32>,
    /// Imaginary part for freq_cis for RoPE relatively positional embeddings (seq_len, dim/2)
    freq_cis_imag: Vec<f32>,
    /// Classifier weights on the logits on the last layer (optional)
    wcls: Vec<f32>,
}

/// State during inference run
struct RunState {
    /// Activation at the current timestamp (dim,)
    x: Vec<f32>,
    /// Activation at the current timestamp inside a residual branch (dim,)
    xb: Vec<f32>,
    /// An additional buffer, for convenience (dim,) (??)
    xb2: Vec<f32>,
    /// Buffer for hidden dimension in the FFN (hidden_dim,)
    hb: Vec<f32>,
    /// Another Buffer for hidden dimension in the FFN (hidden_dim,)
    hb2: Vec<f32>,
    /// Query (dim,)
    q: Vec<f32>,
    /// Key (dim,)
    k: Vec<f32>,
    /// Value (dim,)
    v: Vec<f32>,
    /// Buffer for scores / attention values (n_heads, seq_len)
    att: Vec<f32>,
    /// Output logits (dim?,)
    logits: Vec<f32>,
    /// Key cache (layer, seq_len, dim)
    key_cache: Vec<f32>,
    /// Value cache (layer, seq_len, dim)
    value_cache: Vec<f32>,
}

fn main() {
    println!("Welcome to Llama.rs");
}
