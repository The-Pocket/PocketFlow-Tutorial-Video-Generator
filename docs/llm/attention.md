# **Give Me 20 Minutes, I Will Make Attention Click Forever**

## Introduction: The Core Problem & Our Blueprint

The word "**bank**" in "I sat on the river **bank**" is completely different from the "**bank**" in "I withdrew money from the **bank**." For a machine, this is a huge problem. How can the representation of a word change based on its neighbors?

The answer is the **Attention Mechanism**. In the next 20 minutes, you will understand exactly how it works. You will learn:

*   **Word Embeddings:** The static starting point for every word.
*   **Scaled Dot-Product Attention:** The core formula that enables context.
*   **Query, Key, and Value (QKV):** The three roles a word can play.
*   **The Causal Mask:** How to prevent the model from cheating by looking ahead.
*   **Multi-Head Attention:** How to scale the mechanism for powerful models.

This is the entire secret. This single formula and the Python code that implements it are the engine behind models like ChatGPT.

**The Formula:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

**The Code:**
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```
This might look intimidating, but we will build it from the ground up until every line is obvious. But before we can build this engine, we must first understand the fuel it runs on: word vectors.

---

### **Chapter 1: The Starting Point & Its Fatal Flaw (Word Embeddings)**

A neural network cannot understand the word "cat". It can only understand lists of numbers, called **vectors**. Our first job is to convert every word in our vocabulary into a unique vector.

The mechanism for this is a simple lookup table called an **Embedding Layer**.

**The Mechanism: A Learnable Dictionary**
Imagine a giant spreadsheet with one row for every word in the vocabulary. Each row contains the vector for that word. The `nn.Embedding` layer is exactly this.

```python
import torch
import torch.nn as nn

# A tiny config for our example
vocab_size = 10    # Our dictionary has 10 words
n_embd = 4         # Each word will be represented by a vector of size 4

# The layer is our coordinate book
token_embedding_table = nn.Embedding(vocab_size, n_embd)

# Let's look up the vector for the word with ID=3
input_id = torch.tensor([3])
vector = token_embedding_table(input_id)

print(f"The vector for word ID {input_id.item()} is:\n{vector}")
```
**Output:**
```
The vector for word ID 3 is:
tensor([[-1.5323, -0.2343,  0.5132, -1.0833]], grad_fn=<EmbeddingBackward0>)
```
Initially, these vectors are random. During training, the model learns the optimal vector for each word.

**The Fatal Flaw: No Context**

This simple lookup has one massive problem: it is **static**. The vector for a word is the same regardless of the words around it.

Let's return to our "bank" example.
*   Sentence 1: "I sat on the river **bank**."
*   Sentence 2: "I withdrew money from the **bank**."

Let's assume the word "bank" has ID `7` in our vocabulary. When we look up its vector, the process is identical for both sentences.

| Context | Word | Lookup Process | Resulting Vector |
| :--- | :--- | :--- | :--- |
| "river..." | bank | `embedding_table[7]` | `[0.1, 0.8, -0.4, ...]` |
| "money..." | bank | `embedding_table[7]` | `[0.1, 0.8, -0.4, ...]` **(Identical!)** |

This is the core limitation. Our initial vectors are context-free. They represent a word's general meaning but are blind to the specific meaning in a sentence.

This sets up our central question: **How can we dynamically modify a word's vector using the context from its neighbors?**

The answer is the Attention mechanism, which we will build, step-by-step, in the next chapter.

## **Chapter 2: The Core Idea - A "Conversation" Between Words**

How do we fix the fatal flaw of static embeddings? The vector for "bank" must become more "river-like" or more "money-like" depending on its context.

The solution is to let the words in a sentence communicate with each other. The **Attention** mechanism allows each word to look at its neighbors and create a new, context-aware vector for itself.

**The Analogy: A "Conversation" to Resolve Ambiguity**

Think of this process as a three-step conversation for each word. Let's use the sentence "**Crane** lifted steel." The initial vector for "crane" is ambiguous (bird or machine?). To clarify itself, "crane" will:

1.  **Ask a Question:** It will formulate a query about itself.
2.  **Find Relevant Neighbors:** It will compare its query to labels provided by every other word.
3.  **Absorb Information:** It will take a weighted average of information from its neighbors, listening more to the most relevant ones.

To make this possible, each word's initial vector is used to derive three new vectors: a **Query**, a **Key**, and a **Value**.

| Vector | Role | Analogy |
| :--- | :--- | :--- |
| **Query (Q)** | What I'm looking for. | The word's "search query" or "question." |
| **Key (K)** | What I have. | The word's "label" or "keyword." This is what queries are matched against. |
| **Value (V)**| What I'll give you. | The actual "information" or "substance" the word provides if a match is found. |

**A Concrete Walkthrough: "Crane lifted steel"**

Let's imagine a 2D space where Dimension 1 is "Animal-ness" and Dimension 2 is "Machine-ness".

The ambiguous "crane" starts with a balanced vector. To resolve this, it uses its **Query** to probe the **Keys** of all words in the sentence (including itself).

```
Image description:
Three dots on a 2D plane labeled "Animal-ness" (x-axis) and "Machine-ness" (y-axis).
- A dot for "crane" is at (0.7, 0.7), representing ambiguity.
- A dot for "lifted" is at (0.1, 0.9), highly machine-like.
- A dot for "steel" is at (0.1, 0.9), highly machine-like.
An arrow originates from the "crane" dot, pointing towards the other dots, labeled "Query".
Each dot has a label next to it, "Key".
```

**Step 1: Scoring (Query probes Key)**
The "crane" query, which is looking for context, finds a strong match with the "machine-like" keys of "lifted" and "steel". The mathematical operation for this "matching" is the **dot product**. A high dot product means high similarity.

*   `Score(crane -> lifted)`: HIGH (The machine-like parts align)
*   `Score(crane -> steel)`: HIGH (The machine-like parts align)
*   `Score(crane -> crane)`: Medium (It aligns with its own ambiguous self)

**Step 2: Normalizing (Deciding who to listen to)**
The raw scores are converted into percentages that sum to 1. This is done with the **softmax** function.

*   Attention for "crane" might become: { `crane`: 20%, `lifted`: 40%, `steel`: 40% }
*   This means "crane" has decided to construct its new self by listening mostly to "lifted" and "steel".

**Step 3: Aggregating (Absorbing the information)**
The new vector for "crane" is a weighted average of all the **Value** vectors in the sentence.

*   `New_Vector(crane) = 0.2 * V(crane) + 0.4 * V(lifted) + 0.4 * V(steel)`

Since the Value vectors for "lifted" and "steel" are heavily "machine-like," they pull the new "crane" vector in that direction.

```
Image description:
The same 2D plane as before.
The original "crane" dot is still at (0.7, 0.7) but is now faded.
Two arrows, one from "lifted" and one from "steel", point to a new location on the plane.
A new, solid dot for "crane" now appears at approximately (0.3, 0.8), much higher on the "Machine-ness" axis.
This new dot is labeled "Updated Crane Vector".
```

The result: The original, ambiguous "crane" vector has been transformed into a new, context-aware vector that is unambiguously "machine-like". The conversation worked.

This three-step process—**Score, Normalize, Aggregate**—is the heart of the Attention mechanism. In the next chapter, we will translate this exact logic into efficient matrix operations.

## **Chapter 3: The Engine - Dot-Product Attention in Code**

We've built the intuition: **Score, Normalize, Aggregate**. Now, let's translate this "conversation" into efficient matrix mathematics. By performing these three steps on matrices, we can process every word in a sentence simultaneously.

Our map for this chapter is the core of the attention formula:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

We will build this with raw tensors to see every number. We'll use a tiny example sentence with 3 tokens (`T=3`), each represented by a 2-dimensional vector (`C=2`).

**The Input (`x`): Our Static Embeddings**
This is the tensor from our embedding layer. Shape `(B, T, C)` is `(1, 3, 2)`.

```python
import torch
import torch.nn.functional as F
import math

# Our input: Batch=1, Tokens=3, Channels=2
x = torch.tensor([[[1.0, 0.2],   # Vector for Token 1
                   [0.8, 0.5],   # Vector for Token 2
                   [0.1, 0.9]]]) # Vector for Token 3
```

**Step 1: Get Q, K, and V**
In a real model, Q, K, and V are produced by passing `x` through three separate, learnable `nn.Linear` layers. This allows the model to learn the best "query", "key", and "value" representation for each word. For this tutorial, we will simplify and set them all equal to `x`.

```python
q, k, v = x, x, x
```

**Step 2: Score (`QK^T`)**
This is the heart of the "conversation." To compute the similarity score of every token's query with every other token's key, we use a single matrix multiplication.
*   `q` has shape `(1, 3, 2)`.
*   We transpose `k` to `k.transpose(-2, -1)`, giving it a shape of `(1, 2, 3)`.
*   The multiplication `(1, 3, 2) @ (1, 2, 3)` results in a `(1, 3, 3)` matrix of scores.

```python
scores = q @ k.transpose(-2, -1)
print("--- Raw Scores (Attention Matrix) ---")
print(scores.shape)
print(scores.data.round(decimals=2))
```
**Output:**
```
--- Raw Scores (Attention Matrix) ---
torch.Size([1, 3, 3])
tensor([[[1.04, 0.90, 0.28],   # Token 1's scores for (T1, T2, T3)
         [0.90, 0.89, 0.53],   # Token 2's scores for (T1, T2, T3)
         [0.28, 0.53, 0.82]]])  # Token 3's scores for (T1, T2, T3)
```
This `(3,3)` matrix holds the raw compatibility scores. For example, the query for Token 1 (row 0) has the highest compatibility with the key for Token 1 (column 0), which is `1.04`.

**Step 3: Scale**
This is the ` / sqrt(d_k)` part of the formula. We divide the scores by the square root of the key dimension (`d_k` is the last dimension of `k`, which is `2`). This is a small technical detail that helps stabilize the training process, especially in large models.
```python
d_k = k.size(-1) # d_k = 2
scaled_scores = scores / math.sqrt(d_k)
```

**Step 4: Normalize (`softmax`)**
We apply the `softmax` function along each row. This converts the raw scores into attention weights that sum to 1, representing the percentages from our intuition.
```python
weights = F.softmax(scaled_scores, dim=-1) # Softmax along the rows
print("\n--- Attention Weights ---")
print(weights.data.round(decimals=2))
```
**Output:**
```
--- Attention Weights ---
tensor([[[0.39, 0.35, 0.26],
         [0.36, 0.36, 0.28],
         [0.28, 0.34, 0.38]]])
```
Each row now sums to 1. For example, Token 2 (row 1) will construct its new self by listening 36% to Token 1, 36% to itself, and 28% to Token 3.

**Step 5: Aggregate Values (`weights @ V`)**
Finally, we use our weights to create a weighted average of the **Value** vectors.
*   `weights` has shape `(1, 3, 3)`.
*   `v` has shape `(1, 3, 2)`.
*   The multiplication `(1, 3, 3) @ (1, 3, 2)` produces a final tensor of shape `(1, 3, 2)`.

```python
output = weights @ v
print("\n--- Final Output (Context-Aware Vectors) ---")
print(output.shape)
print(output.data.round(decimals=2))
```
**Output:**
```
--- Final Output (Context-Aware Vectors) ---
torch.Size([1, 3, 2])
tensor([[[0.69, 0.51],
         [0.67, 0.53],
         [0.59, 0.58]]])
```
Success! We have taken our raw input `x` and produced a new tensor `output` of the exact same shape, where each token's vector has been updated with information from its neighbors.

Here is a summary of the tensor transformations:

| Step | Operation | Input Shapes | Output Shape `(B, T, ...)` |
| :--- | :--- | :--- | :--- |
| 1 | `Q, K, V = proj(x)` | `(1, 3, 2)` | `(1, 3, 2)` |
| 2 | `Q @ K.T` | `(1, 3, 2)` & `(1, 2, 3)` | `(1, 3, 3)` |
| 3 | `/ sqrt(d_k)` | `(1, 3, 3)` | `(1, 3, 3)` |
| 4 | `softmax` | `(1, 3, 3)` | `(1, 3, 3)` |
| 5 | `weights @ V`| `(1, 3, 3)` & `(1, 3, 2)` | `(1, 3, 2)` |

We have now built the core engine. In the next chapter, we'll add two crucial upgrades to make it practical for real-world models.

## **Chapter 4: The Upgrades - Making Attention Practical**

We have built the core attention engine. However, to use it in a real model like GPT, we need two crucial upgrades.
1.  **Causality:** We must prevent the model from looking into the future when generating text.
2.  **Parallelism:** We need to make the "conversation" richer by allowing it to happen from multiple perspectives at once.

#### **Part 1: The Causal Mask ("Don't Look Ahead")**

**The Problem:** GPT is an **autoregressive** model. When predicting the next word in the sentence "A cat sat...", its decision must be based *only* on the tokens it has seen so far: "A" and "cat". It cannot be allowed to see the answer, "sat".

Our current attention matrix allows this cheating. The token "A" (at position 0) is gathering information from "cat" (position 1) AND "sat" (position 2). This is a problem.

**The Solution:** The Causal Mask. We will modify the attention **score matrix** *before* applying the softmax function. We will "mask out" all future positions by setting their scores to negative infinity (`-inf`).

Why `-inf`? Because the `softmax` function involves an exponential: `e^x`. The exponential of negative infinity, `e^-inf`, is effectively zero. This forces the attention weights for all future tokens to become `0`, preventing any information flow.

**The Mechanism:**
1.  **Create a Mask:** We use `torch.tril` to create a lower-triangular matrix. The `0`s in the upper-right triangle represent the "future" connections we must block.
    ```python
    T = 3
    mask = torch.tril(torch.ones(T, T))
    print("--- The Mask ---")
    print(mask)
    # tensor([[1., 0., 0.],
    #         [1., 1., 0.],
    #         [1., 1., 1.]])
    ```
2.  **Apply the Mask:** We use `masked_fill` to apply our mask to the `scaled_scores` from the last chapter.
    ```python
    # Before masking
    # scaled_scores = tensor([[[0.74, 0.64, 0.20], ... ]])

    masked_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

    print("\n--- Scores After Masking ---")
    print(masked_scores.data.round(decimals=2))
    # tensor([[[ 0.74, -inf, -inf],
    #          [ 0.64,  0.63, -inf],
    #          [ 0.20,  0.37,  0.58]]])
    ```
3.  **Re-run Softmax:** Applying softmax to these masked scores gives us causal attention weights.
    ```python
    causal_weights = F.softmax(masked_scores, dim=-1)
    print("\n--- Final Causal Attention Weights ---")
    print(causal_weights.data.round(decimals=2))
    # tensor([[[1.00, 0.00, 0.00],
    #          [0.50, 0.50, 0.00],
    #          [0.29, 0.35, 0.36]]])
    ```
The upper-right triangle of our attention matrix is now all zeros. "A" can only attend to itself. "cat" can only attend to "A" and itself. Information now only flows from the past to the present.

---
#### **Part 2: Multi-Head Attention ("Many Conversations at Once")**

**The Problem:** Our current attention mechanism is like having one person in a meeting who is responsible for figuring out all the relationships between words (syntax, semantics, etc.). This is a lot of pressure.

**The Solution:** Multi-Head Attention. We split our embedding dimension `C` into several smaller chunks, called "heads". Each head will be its own independent attention mechanism, conducting its own "conversation" in parallel.

*   **Head 1** might learn to focus on verb-object relationships.
*   **Head 2** might learn to focus on pronoun references.
*   ...and so on.

**The Mechanism:**
Let's use a realistic `C = 768` and `n_head = 12`. The dimension of each head will be `head_dim = C / n_head = 64`.

1.  **Split:** We take our Q, K, and V tensors (each shape `B, T, C`) and reshape them to `(B, n_head, T, head_dim)`. This makes the "heads" an explicit dimension.
    ```python
    # B=1, T=3, C=768
    q = torch.randn(1, 3, 768)
    
    # Split C into (n_head, head_dim) -> (12, 64)
    q_multi_head = q.view(1, 3, 12, 64)
    
    # Bring the head dimension forward for parallel computation
    q_multi_head = q_multi_head.transpose(1, 2) # -> (1, 12, 3, 64)
    ```
2.  **Attend in Parallel:** We perform the exact same scaled dot-product attention as before. PyTorch's broadcasting automatically handles the `n_head` dimension, performing 12 attention calculations at once. The output has shape `(B, n_head, T, head_dim)`.
3.  **Merge:** We reverse the split operation. We concatenate the heads back together into a single `C`-dimensional vector.
    ```python
    # Transpose back and reshape
    merged_output = output_per_head.transpose(1, 2).contiguous().view(1, 3, 768)
    ```
4.  **Project:** We pass this merged output through a final linear layer (`c_proj`). This allows the model to learn how to best combine the insights from all the different heads.

By having multiple parallel conversations, the model can analyze the input text from many different perspectives at the same time, making it far more powerful.

We now have all the conceptual pieces. In the final chapter, we will assemble them into our complete, production-ready code.

## **Chapter 5: The Final Blueprint & Conclusion**

We have built all the pieces: the core engine, the causal mask, and the multi-head architecture. Now, let's look at our final blueprint one last time to see how these concepts snap together into a single, elegant piece of code.

The intimidating module from the introduction should now look like a familiar map.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # The layers for QKV projection, multi-head output, and the mask
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        # ...

    def forward(self, x):
        B, T, C = x.size()
        
        # 1. Get Q, K, V from a single efficient projection
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 2. Split into multiple heads for parallel conversations
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. The core engine: scaled, masked, dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # No looking ahead!
        att = F.softmax(att, dim=-1)
        y = att @ v

        # 4. Merge the heads back together and finalize
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```
Each part of this code now has a clear purpose:
*   **Problem:** We take a static input `x` and produce a context-aware output `y`.
*   **Intuition:** The Q, K, V "conversation" is implemented here.
*   **Engine:** The core `q @ k.transpose...` logic is the mathematical heart.
*   **Upgrades:** The `masked_fill` provides causality, and the `view/transpose` operations create the parallel heads.

In the last 20 minutes, we have gone on a journey. We started with a fundamental problem: words are static, but meaning depends on context. We solved it by building a mechanism that allows words to have a "conversation."

We built the intuition for this conversation with Queries, Keys, and Values. We translated that intuition into the efficient mathematics of dot-product attention. We then upgraded it with a causal mask and multi-head parallelism to make it powerful and practical.

This `CausalSelfAttention` module is the single most important component of modern large language models. It is the engine that drives understanding in every "Transformer Block," which are then stacked dozens of times to create models like GPT.

The magic is gone, replaced by elegant, understandable engineering. **Attention has clicked.**