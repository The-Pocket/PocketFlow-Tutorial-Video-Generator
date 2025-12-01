# **Title: Rotary Positional Encoding (RoPE): A Deep Dive**

## **Introduction: Give me 40 minutes, and I will make RoPE click forever.**

You have likely heard that modern Large Language Models like Llama, PaLM, and GPT-NeoX have abandoned traditional positional embeddings. Instead, they use a more powerful and elegant method: **Rotary Positional Encoding**, or **RoPE**.

What if I told you the entire "magic" behind this technique is just high-school trigonometry? What if the core of this powerful idea is captured in this single matrix operation?

$$
\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix} = \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0) \\ \sin(m\theta_0) & \cos(m\theta_0) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
$$

This is the 2D rotation matrix. It's the entire secret. By the end of this tutorial, this formula will not only make sense, but you will understand how it's generalized to high dimensions and why it's the key to unlocking **relative positional information** in Transformers.

#### **Our Promise**

Our promise is simple: in the next 40 minutes, you will understand RoPE from first principles to practical implementation. You will understand not only **what** the algorithm is, but **why** it works so effectively. By the end, you will understand this code:

```python
def apply_rotary_pos_emb(x: torch.Tensor, rope_emb: torch.Tensor):
    seq_len = x.shape[1]
    rope_emb_sliced = rope_emb[:seq_len, :].unsqueeze(0).unsqueeze(2)
    cos_emb = rope_emb_sliced.cos()
    sin_emb = rope_emb_sliced.sin()

    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    x_partner = x_partner.flatten(-2)

    return (x * cos_emb + x_partner * sin_emb).type_as(x)
```

This is the complete RoPE implementation. Every line will make sense by the end.

## **Chapter 1: The Problem: The Limitations of Absolute Positions**

To understand why RoPE is a breakthrough, we must first understand the method it replaced: **Absolute Positional Embeddings**.

#### **The Old Way: Assigning an "Address" to Each Position**

In early Transformer models like BERT and GPT-2, the position of a token was handled by creating a unique vector for each possible position, up to a maximum length (e.g., 1024).

1.  A token's meaning is represented by its **Token Embedding**.
2.  A token's location is represented by its **Positional Embedding**.
3.  The final input vector is the sum: `Input Vector = Token Embedding + Positional Embedding`.

This is like giving each word in a sentence a specific street address.

| Word | Token Embedding | Position | Positional Embedding | Final Input Vector |
| :--- | :--- | :--- | :--- | :--- |
| "The" | `vec("The")` | 0 | `vec(pos=0)` | `vec("The") + vec(pos=0)` |
| "red" | `vec("red")` | 1 | `vec(pos=1)` | `vec("red") + vec(pos=1)` |
| "car" | `vec("car")` | 2 | `vec(pos=2)` | `vec("car") + vec(pos=2)` |

#### **The Flaw: Context is Relative, but Addresses are Absolute**

Language is built on relative relationships. The meaning of "red car" doesn't change based on where it appears in a document. However, the absolute embedding method fundamentally changes the input vectors.

Consider these two sentences:
1.  "**The red car** is fast."
2.  "I saw **the red car**."

Let's look at the final vector for the word "red" in each sentence.

*   In sentence 1, "red" is at position 1. Its final vector is `vec("red") + vec(pos=1)`.
*   In sentence 2, "red" is at position 3. Its final vector is `vec("red") + vec(pos=3)`.

These are two different vectors. The model receives a different input for the exact same word, simply because its absolute position changed. The attention mechanism now has a harder job. It must learn from scratch that the relationship between `vec("red") + vec(pos=1)` and `vec("car") + vec(pos=2)` is the same as the relationship between `vec("red") + vec(pos=3)` and `vec("car") + vec(pos=4)`.

The model doesn't inherently know that "position 4" is one step away from "position 3". It only knows that `vec(pos=3)` and `vec(pos=4)` are two distinct, arbitrary vectors that it needs to learn the relationship between. This is computationally expensive and doesn't generalize well to positions the model hasn't seen during training.

#### **The Goal: A New System Based on Relative Distance**

We need an encoding scheme that bakes the concept of relative position directly into the math. Ideally, the attention score between a query vector `q` at position `m` and a key vector `k` at position `n` should be computable from a function that looks like this:

`Score = f(q, k, m-n)`

The score should depend on the vectors themselves and their **relative distance `m-n`**, not their absolute positions `m` and `n`.

This is the problem RoPE solves. It provides a way to modify `q` and `k` such that their dot product naturally produces this desired relative relationship. The solution, as we will see, is found not in adding vectors, but in rotating them.

Before we dive in, let's clarify exactly how RoPE fits into the Transformer:

| Question | Answer |
|:---|:---|
| **Applied to which vectors?** | Only Q and K, not V. We need position in the attention score (`QK^T`), not in the output values. |
| **Is RoPE itself learned?** | No. The rotation angles are fixed formulas based on position. But the Q and K projection weights (Wq, Wk) *are* learned, and they learn to produce vectors that work well with these rotations. |
| **Applied in every layer?** | Yes. Every transformer block applies RoPE to its Q and K vectors independently. |

## **Chapter 2: The Core Intuition: Encoding Position via Rotation (in 2D)**

We need a transformation that modifies a vector to encode its position while preserving its original information (its meaning). The key insight of RoPE is that a **rotation** does exactly this.

A rotation changes a vector's direction but, crucially, **it does not change its length (norm)**. We can use the original length to represent the token's meaning and the new direction to represent its position.

#### **The Building Block: 2D Rotation**

Let's start in two dimensions. Imagine a token's meaning is captured by a simple 2D vector, `v = (x, y)`. To rotate this vector by an angle `θ`, we multiply it by the standard 2D rotation matrix:

$$
R(\theta) = \begin{pmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{pmatrix}
$$

The new, rotated vector `v'` is calculated as:

$$
v' = R(\theta)v = \begin{pmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}
$$

#### **A Concrete Example: Rotation by 90 Degrees**

Let's make this tangible. Suppose our vector is `v = (1, 2)` and we want to rotate it by `θ = 90°` (or `π/2` radians).

1.  **The Angle:** We know `cos(90°) = 0` and `sin(90°) = 1`.

2.  **The Rotation Matrix:** Plugging these values into `R(θ)` gives us:
    $$
    R(90^{\circ}) = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
    $$

3.  **The Calculation:** Now we perform the matrix multiplication:
    $$
    v' = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} (0 \cdot 1) + (-1 \cdot 2) \\ (1 \cdot 1) + (0 \cdot 2) \end{pmatrix} = \begin{pmatrix} -2 \\ 1 \end{pmatrix}
    $$
    Our new vector is `v' = (-2, 1)`.

4.  **Verifying the Length:**
    *   Length of original vector `v`: `||v|| = sqrt(1² + 2²) = sqrt(5)`
    *   Length of rotated vector `v'`: `||v'|| = sqrt((-2)² + 1²) = sqrt(4 + 1) = sqrt(5)`
    The length is perfectly preserved. All we changed was the direction.

```
A 2D coordinate plane.
The x-axis goes from -3 to 3. The y-axis goes from -3 to 3.
Vector v starts at the origin (0,0) and points to the dot (1, 2). It's an arrow in the first quadrant.
Vector v' starts at the origin (0,0) and points to the dot (-2, 1). It's an arrow in the second quadrant.
An arc with an arrow shows the counter-clockwise 90-degree rotation from v to v'.
```

#### **The "Aha!" Moment: Connecting Rotation to Position**

Here is the core idea of RoPE: **The angle of rotation is determined by the token's position `m`**.

We define a constant, base "frequency" or angle, `θ`. The total rotation applied to a vector at position `m` is simply `m * θ`.

| Position (`m`) | Total Rotation Angle | What it means |
| :--- | :--- | :--- |
| 0 | `0 * θ = 0` | The vector for the first token is **not rotated**. It is our baseline. |
| 1 | `1 * θ = θ` | The vector for the second token is rotated by a small angle `θ`. |
| 2 | `2 * θ` | The vector for the third token is rotated by twice that angle. |
| 3 | `3 * θ` | The vector for the fourth token is rotated by three times that angle. |

This establishes a clear, consistent rule: the further a token is in the sequence, the more its vector is "spun" around the origin.

```
A 3D visualization.
The X and Y axes form a 2D plane at the bottom. The Z axis represents the position 'm' and goes upwards.
At z=0 (position 0), a vector 'v' points from the origin along the positive X-axis.
At z=1 (position 1), the same vector is shown, but rotated slightly counter-clockwise in the XY plane.
At z=2 (position 2), the vector is rotated even more.
At z=3 (position 3), it's rotated further still.
A dotted line connects the tips of these vectors, forming a spiral or helix shape that winds upwards along the Z-axis.
This image shows how the vector's direction in the XY plane progressively changes as its position 'm' increases.
```

We have now established the fundamental principle in a simple 2D world. The next challenge is to figure out how to apply this "rotation" concept to the high-dimensional vectors (e.g., 768 or 4096 dimensions) that are actually used in Large Language Models.

## **Chapter 3: Scaling to High Dimensions: The RoPE Algorithm**

We have a solid principle for 2D vectors, but in a real Transformer, our Query and Key vectors have high dimensions (`d`), for example, `d=128` for a single attention head. How do we "rotate" a 128-dimensional vector?

A single rotation matrix for 128 dimensions would be enormous and complex. RoPE uses a much simpler and more elegant approach.

#### **The Solution: Many Small, Independent Rotations**

Instead of one big rotation, we perform many small 2D rotations. The core trick is to **group the dimensions of the vector into pairs.**

For a vector `x` with `d` dimensions, `x = (x_0, x_1, x_2, x_3, ..., x_{d-2}, x_{d-1})`, we form `d/2` pairs:
*   Pair 0: `(x_0, x_1)`
*   Pair 1: `(x_2, x_3)`
*   ...
*   Pair `i`: `(x_{2i}, x_{2i+1})`
*   ...
*   Final Pair: `(x_{d-2}, x_{d-1})`

We then apply our 2D rotation to **each of these pairs independently**.

#### **The Second Trick: Different Speeds of Rotation**

Think about how a clock tells time. It has three hands that all rotate, but at different speeds:
- The **second hand** rotates fast - one full circle per minute
- The **minute hand** rotates slower - one full circle per hour
- The **hour hand** rotates slowest - one full circle per 12 hours

Why do we need all three? Because a single hand would be ambiguous. If you only had a second hand, you couldn't tell 1:00 from 2:00 - the hand would be in the same position. But the *combination* of all three hands at different speeds gives every moment a unique signature. And crucially, the *difference* between two times is easy to read - if the minute hand moved 5 ticks, 5 minutes passed, regardless of what hour it is.

RoPE works exactly the same way, but instead of 3 hands, we have `d/2` hands (64 for a 128-dimensional vector). Each dimension pair is like a clock hand rotating at its own speed:

*   The first pairs rotate **quickly** - like a second hand, sensitive to nearby positions
*   The last pairs rotate **very slowly** - like an hour hand, tracking long-range position

The rotation speed `θ_i` for the `i`-th pair is:
$$
\theta_i = 10000^{-\frac{2i}{d}}
$$

For a 128-dimensional vector, this gives us 64 "clock hands" with periods ranging from ~6 tokens (fastest) to ~60,000 tokens (slowest). The combination creates a unique signature for each position, and the relative distance between positions is naturally encoded in how much each hand has moved.

#### **The Full RoPE Algorithm**

We can now formalize the complete algorithm for applying RoPE to a single vector $x$ at position $m$.

**Given:**
- A vector $x = (x_0, x_1, x_2, x_3, \ldots, x_{d-1})$ of dimension $d$
- A position $m$

**Algorithm:** For each pair $i$ from $0$ to $\frac{d}{2} - 1$:

1. Calculate the frequency:
$$\theta_i = 10000^{-\frac{2i}{d}}$$

2. Calculate the rotation angle:
$$\alpha = m \cdot \theta_i$$

3. Apply the 2D rotation to the $i$-th pair:
$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

Return the rotated vector $x' = (x'_0, x'_1, \ldots, x'_{d-1})$.

**Quick Example:** Let's apply RoPE to a 4-dimensional vector $x = (1.0, 0.5, 0.8, 0.3)$ at position $m = 2$.

**Pair 0:** $(x_0, x_1) = (1.0, 0.5)$
- $\theta_0 = 10000^{0} = 1.0$
- $\alpha = 2 \times 1.0 = 2.0$
- $\cos(2.0) = -0.42, \quad \sin(2.0) = 0.91$
$$
\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix} = \begin{pmatrix} -0.42 & -0.91 \\ 0.91 & -0.42 \end{pmatrix} \begin{pmatrix} 1.0 \\ 0.5 \end{pmatrix} = \begin{pmatrix} -0.42 \times 1.0 + (-0.91) \times 0.5 \\ 0.91 \times 1.0 + (-0.42) \times 0.5 \end{pmatrix} = \begin{pmatrix} -0.87 \\ 0.70 \end{pmatrix}
$$

**Pair 1:** $(x_2, x_3) = (0.8, 0.3)$
- $\theta_1 = 10000^{-0.5} = 0.01$
- $\alpha = 2 \times 0.01 = 0.02$
- $\cos(0.02) = 1.00, \quad \sin(0.02) = 0.02$
$$
\begin{pmatrix} x'_2 \\ x'_3 \end{pmatrix} = \begin{pmatrix} 1.00 & -0.02 \\ 0.02 & 1.00 \end{pmatrix} \begin{pmatrix} 0.8 \\ 0.3 \end{pmatrix} = \begin{pmatrix} 1.00 \times 0.8 + (-0.02) \times 0.3 \\ 0.02 \times 0.8 + 1.00 \times 0.3 \end{pmatrix} = \begin{pmatrix} 0.79 \\ 0.32 \end{pmatrix}
$$

**Result:** $x' = (-0.87, 0.70, 0.79, 0.32)$

Notice: Pair 0 rotated significantly (fast clock hand), while Pair 1 barely moved (slow clock hand).

This is the entire forward pass of RoPE. It's a deterministic transformation applied to the Query and Key vectors. In the next chapter, we will prove mathematically why this elegant procedure results in the exact relative positioning property we set out to achieve.

## **Chapter 4: The Mathematical Proof: Why RoPE is Relative**

We've built the RoPE algorithm. Now it's time for the payoff. We will prove that this method of rotating vector pairs creates the exact relative positioning property we wanted.

**Why the Dot Product?** Recall how attention works: `Attention = softmax(QK^T)V`. The `QK^T` part computes dot products between every query and key. This dot product determines how much one token attends to another. So if we want relative position information to influence attention, we need it to show up in this dot product.

**Our Goal:** To show that after applying RoPE, this dot product depends only on the original vectors and their *relative* distance `m-n`, not their absolute positions.

Formally, we want to prove there exists a function `g` such that:
$$
\text{RoPE}(q, m)^T \cdot \text{RoPE}(k, n) = g(q, k, m-n)
$$

The absolute positions `m` and `n` should disappear, leaving only their difference.

Since RoPE treats each pair of dimensions independently, we only need to prove this for a single 2D pair. The result will hold for the sum of dot products across all pairs, and thus for the full high-dimensional vectors.

#### **Setup for a Single 2D Pair**

Let's consider a single pair of dimensions for our query and key vectors.
*   The query pair at position `m`: `q_m = (q_0, q_1)`
*   The key pair at position `n`: `k_n = (k_0, k_1)`
*   The rotation frequency for this pair: `θ` (we'll drop the subscript `i` for clarity).

**Step 1: Apply RoPE to the Query**
We apply the rotation for position `m`. The rotated query `q'_m` is:
$$
q'_m = R(m\theta)q_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
$$

**Step 2: Apply RoPE to the Key**
We apply the rotation for position `n`. The rotated key `k'_n` is:
$$
k'_n = R(n\theta)k_n = \begin{pmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{pmatrix} \begin{pmatrix} k_0 \\ k_1 \end{pmatrix}
$$

**Step 3: Calculate the Dot Product**
The attention score for this pair is their dot product, `(q'_m)^T (k'_n)`. Let's write this out using our matrix expressions:
$$
\text{Score} = (R(m\theta)q_m)^T (R(n\theta)k_n)
$$
Using the transpose property `(AB)^T = B^T A^T`, we get:
$$
\text{Score} = q_m^T R(m\theta)^T R(n\theta) k_n
$$

This is the crucial step. We need to simplify the product of the two rotation matrices in the middle.

**Step 4: The Key Property of Rotation Matrices**
Rotation matrices have a beautiful property: the transpose of a rotation matrix is the same as the matrix for the inverse rotation.
$$
R(\alpha)^T = R(-\alpha) = \begin{pmatrix} \cos(-\alpha) & -\sin(-\alpha) \\ \sin(-\alpha) & \cos(-\alpha) \end{pmatrix} = \begin{pmatrix} \cos(\alpha) & \sin(\alpha) \\ -\sin(\alpha) & \cos(\alpha) \end{pmatrix}
$$
Another property is that multiplying two rotation matrices is the same as adding their angles: `R(α)R(β) = R(α+β)`.

Let's apply these properties to our term `R(mθ)^T R(nθ)`:
$$
R(m\theta)^T R(n\theta) = R(-m\theta) R(n\theta) = R(n\theta - m\theta) = R((n-m)\theta)
$$

**The "Aha!" Moment**
The product of the two rotation matrices simplifies to a single rotation matrix whose angle is determined by the **relative distance `n-m`**.

**Step 5: The Final Result**
Now we can substitute this back into our score equation:
$$
\text{Score} = q_m^T R((n-m)\theta) k_n
$$
Let's expand this to see it clearly:
$$
\text{Score} = \begin{pmatrix} q_0 & q_1 \end{pmatrix} \begin{pmatrix} \cos((n-m)\theta) & -\sin((n-m)\theta) \\ \sin((n-m)\theta) & \cos((n-m)\theta) \end{pmatrix} \begin{pmatrix} k_0 \\ k_1 \end{pmatrix}
$$
This expression depends only on:
1.  The original query components `(q_0, q_1)`.
2.  The original key components `(k_0, k_1)`.
3.  The relative distance `n-m`.

The absolute positions `m` and `n` have vanished from the final equation, replaced entirely by their difference. This is exactly what we set out to prove.

We have now built the mathematical foundation. In the final chapter, we will translate this theory into efficient PyTorch code.

## **Chapter 5: Implementation: Building RoPE from Scratch**

We've explored the theory and proven the mathematics. Now, let's translate the RoPE algorithm into clean, efficient PyTorch code. Our goal is to create a reusable function that can be easily plugged into any Transformer architecture.

The implementation has two main parts:
1.  **Pre-computation:** Calculating the `sin` and `cos` values for all possible positions and frequencies ahead of time. This is a one-time setup cost.
2.  **Application:** Applying these pre-computed rotations to the Query and Key tensors during the model's forward pass.

#### **Snippet 1: Pre-computing the Frequencies and Rotations**

We don't want to re-calculate `sin(mθ_i)` and `cos(mθ_i)` on the fly for every token in every forward pass. It's far more efficient to compute these values once and store them.

Let's write a function that prepares a tensor containing all the necessary rotation angles.

```python
import torch

def precompute_rope_embeddings(head_dim: int, max_seq_len: int, base: int = 10000):
    # 1. Calculate the frequencies (theta_i) for each dimension pair
    # For head_dim=4: torch.arange(0,4,2) = [0, 2]
    #                 exponents = [0/4, 2/4] = [0, 0.5]
    #                 inv_freq = 10000^(-[0, 0.5]) = [1.0, 0.01]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # 2. Create the position index tensor
    # For max_seq_len=4: t = [0, 1, 2, 3]
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # 3. Outer product: each position m multiplied by each frequency theta_i
    # freqs[m, i] = m * theta_i
    # For our example: [[0*1.0, 0*0.01], [1*1.0, 1*0.01], [2*1.0, 2*0.01], [3*1.0, 3*0.01]]
    #                = [[0, 0], [1, 0.01], [2, 0.02], [3, 0.03]]
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    
    # 4. Duplicate frequencies for both elements in each pair
    # [θ0, θ1] -> [θ0, θ0, θ1, θ1] so consecutive elements share the same angle
    # Shape: (max_seq_len, head_dim)
    emb = freqs.repeat_interleave(2, dim=-1)

    return emb

# --- Example Usage ---
MAX_LEN = 4
HEAD_DIM = 4
emb = precompute_rope_embeddings(HEAD_DIM, MAX_LEN)

print("Angles (m * theta_i) for each position and dimension pair:")
print(emb)
print("\nCosines:")
print(emb.cos())
print("\nSines:")
print(emb.sin())
```
**Output:**
```
Angles (m * theta_i) for each position and dimension pair:
tensor([[0.0000, 0.0000, 0.0000, 0.0000],   # position 0: no rotation
        [1.0000, 1.0000, 0.0100, 0.0100],   # position 1: pair 0 gets θ=1.0, pair 1 gets θ=0.01
        [2.0000, 2.0000, 0.0200, 0.0200],   # position 2: 2x the angles
        [3.0000, 3.0000, 0.0300, 0.0300]])  # position 3: 3x the angles

Cosines:
tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
        [ 0.5403,  0.5403,  0.9999,  0.9999],
        [-0.4161, -0.4161,  0.9998,  0.9998],
        [-0.9900, -0.9900,  0.9996,  0.9996]])

Sines:
tensor([[0.0000, 0.0000, 0.0000, 0.0000],
        [0.8415, 0.8415, 0.0100, 0.0100],
        [0.9093, 0.9093, 0.0200, 0.0200],
        [0.1411, 0.1411, 0.0300, 0.0300]])
```
Notice how pair 0 (columns 0,1) rotates quickly - by position 2 the cosine is already negative (-0.4161). Pair 1 (columns 2,3) barely moves - cosine stays near 1.0. This is our "clock hands at different speeds."

#### **Snippet 2: Applying the Rotations**

Now for the core function. This function will take a Query or Key tensor and apply the pre-computed rotations. The implementation uses a clever trick to handle the 2D rotations on paired dimensions without any explicit loops.

**The Rotation Trick**

Recall the 2D rotation formulas:
$$
x'_0 = x_0 \cos\theta - x_1 \sin\theta
$$
$$
x'_1 = x_0 \sin\theta + x_1 \cos\theta
$$

We can rewrite this as element-wise operations:
$$
\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix} = \begin{pmatrix} x_0 \\ x_1 \end{pmatrix} \odot \begin{pmatrix} \cos\theta \\ \cos\theta \end{pmatrix} + \begin{pmatrix} -x_1 \\ x_0 \end{pmatrix} \odot \begin{pmatrix} \sin\theta \\ \sin\theta \end{pmatrix}
$$

The trick: create a "partner" vector by swapping and negating: $(x_0, x_1) \rightarrow (-x_1, x_0)$. Then:
$$
x' = x \odot \cos + x_{\text{partner}} \odot \sin
$$

This allows us to perform all rotations in parallel with efficient tensor operations.

```python
def apply_rotary_pos_emb(x: torch.Tensor, rope_emb: torch.Tensor):
    # Get the sequence length from the input tensor
    seq_len = x.shape[1]
    
    # Slice the pre-computed embeddings to match the sequence length
    # Shape: (1, seq_len, 1, head_dim) for broadcasting
    rope_emb_sliced = rope_emb[:seq_len, :].unsqueeze(0).unsqueeze(2)
    
    # Get the cosine and sine components
    cos_emb = rope_emb_sliced.cos()
    sin_emb = rope_emb_sliced.sin()
    
    # --- The Rotation Trick ---
    # 1. Reshape x to handle pairs
    # (batch, seq_len, num_heads, head_dim/2, 2)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    
    # 2. Create the partner vector: (-x_1, x_0, -x_3, x_2, ...)
    # x_partner's first element is -x_reshaped's second element, and vice-versa
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[..., 0]], dim=-1)
    
    # 3. Flatten back to original shape for multiplication
    # (batch, seq_len, num_heads, head_dim)
    x_partner = x_partner.flatten(-2)
    
    # 4. Perform the rotation using element-wise multiplication
    rotated_x = x * cos_emb + x_partner * sin_emb
    
    return rotated_x.type_as(x)

# --- Example Usage ---
# Small example: 1 batch, 2 positions, 1 head, 4 dimensions
rope_emb = precompute_rope_embeddings(head_dim=4, max_seq_len=4)
x = torch.tensor([[
    [[1.0, 0.0, 1.0, 0.0]],  # position 0: pairs (1,0) and (1,0)
    [[1.0, 0.0, 1.0, 0.0]],  # position 1: same vector, different position
]])  # shape: (1, 2, 1, 4)

rotated = apply_rotary_pos_emb(x, rope_emb)
print("Input x:\n", x)
print("\nRotated x:\n", rotated)
```
**Output:**
```
Input x:                          # Same vector at both positions
 tensor([[[[1., 0., 1., 0.]],     # position 0
          [[1., 0., 1., 0.]]]])   # position 1

Rotated x:
 tensor([[[[1.0000, 0.0000, 1.0000, 0.0000]],   # position 0: no rotation (m=0)
          [[0.5403, 0.8415, 0.9999, 0.0100]]]]) # position 1: rotated!
```
At position 0, no rotation (angle = 0). At position 1:
- Pair 0 `(1,0)` rotated by θ=1.0 → `(cos(1), sin(1))` = `(0.54, 0.84)`
- Pair 1 `(1,0)` rotated by θ=0.01 → `(cos(0.01), sin(0.01))` ≈ `(1.0, 0.01)` (barely moved)

The same input vector produces different outputs based on position. That's RoPE! The `apply_rotary_pos_emb` function can now be called on the Query and Key tensors inside the attention block, right before the dot-product is calculated.

#### **Connecting to Reality: Llama-3**
Let's see how our example numbers connect to a real, state-of-the-art model like Llama-3 8B.
*   **Embedding Dimension (`n_embd`):** 4096
*   **Number of Heads (`n_head`):** 32
*   **Head Dimension (`head_dim`):** `4096 / 32 = 128`. This is the `H` in our code.
*   **Max Sequence Length (`block_size`):** 8192. This is the `MAX_LEN`.

When Llama-3 processes a sequence, it takes each of its 32 query heads (each a 128-dimensional vector) and applies exactly the rotation logic we just implemented. It does the same for the key heads. This simple, elegant rotation is a cornerstone of its ability to process long and complex contexts.

---

## **Conclusion: From Absolute Addresses to Relative Directions**

We began our journey with a problem: absolute positional embeddings are rigid and don't naturally capture the relative nature of language. Over the last 40 minutes, we have completely demystified the solution used by today's most powerful models.

*   We built the core intuition that **position can be encoded via rotation**, starting in a simple 2D plane.
*   We scaled this idea to high dimensions by **rotating pairs of dimensions at different speeds**, allowing the model to capture relative distances at multiple scales.
*   We proved mathematically that this rotation scheme guarantees the attention score between two tokens is **a function of their relative distance**, not their absolute locations.
*   Finally, we translated this theory into a **concrete and efficient PyTorch implementation**, ready to be used in a real Transformer.

RoPE is a perfect illustration of a powerful principle in machine learning: injecting a correct and useful **inductive bias** into a model's architecture. Instead of forcing the model to learn the concept of "relative distance" from scratch, we built the idea directly into its geometry. The result is a more efficient, powerful, and flexible model that has become a cornerstone of modern AI. The magic has dissolved into elegant, understandable mathematics.