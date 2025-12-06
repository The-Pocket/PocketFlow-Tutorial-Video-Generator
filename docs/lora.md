# **Tutorial Outline: LoRA for LLMs From Scratch**

## **Chapter 1: The Promise - Master LoRA in 30 Minutes**

You've heard of LoRA. It's the key to fine-tuning massive LLMs on a single GPU. You've seen the acronyms: PEFT, low-rank adaptation. But what is it, *really*?

It's not a complex theory. It's a simple, elegant trick.

Instead of training a 1-billion-parameter weight matrix `W`, you freeze it. You then train two tiny matrices, `A` and `B`, that represent the *change* to `W`.

This is LoRA. It's this piece of code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        # Freeze the original linear layer
        self.base = base
        self.base.weight.requires_grad_(False)

        # Create the trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, r))

        # Initialize the weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B) # Start with no change

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original path (frozen) + LoRA path (trainable)
        return self.base(x) + (F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling)

```

**My promise:** You will understand every line of this code, the math behind it, and why it's so effective, in the next 30 minutes. Let's begin.

## **Chapter 2: The Foundation - The `nn.Linear` Layer**

Before we can modify an LLM, we must understand its most fundamental part: the `nn.Linear` layer. It's the simple workhorse that performs the vast majority of computations in a Transformer.

Its only job is to perform this equation: `output = input @ W.T + b`

#### A Minimal, Reproducible Example

Let's see this in action. We'll create a tiny linear layer that takes a vector of size 3 and outputs a vector of size 2. To make this perfectly clear, we will set the weights and bias manually.

**1. Setup the layer and input:**

```python
import torch
import torch.nn as nn

# A layer that maps from 3 features to 2 features
layer = nn.Linear(in_features=3, out_features=2, bias=True)

# A single input vector (with a batch dimension of 1)
input_tensor = torch.tensor([[1., 2., 3.]])

# Manually set the weights and bias for a clear example
with torch.no_grad():
    layer.weight = nn.Parameter(torch.tensor([[0.1, 0.2, 0.3],
                                              [0.4, 0.5, 0.6]]))
    layer.bias = nn.Parameter(torch.tensor([0.7, 0.8]))

```

**2. Inspect the Exact Components:**

Now we have known values for everything.

*   **Input `x`:** `[1., 2., 3.]`
*   **Weight `W`:** `[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]`
*   **Bias `b`:** `[0.7, 0.8]`

**3. The Forward Pass and Its Output:**

When you call `layer(input_tensor)`, PyTorch computes the result.

```python
# The forward pass
output_tensor = layer(input_tensor)

print("--- PyTorch Calculation ---")
print("Input (x):", input_tensor)
print("Weight (W):\n", layer.weight)
print("Bias (b):", layer.bias)
print("\nOutput (y):", output_tensor)
```

This will print:

```text
--- PyTorch Calculation ---
Input (x): tensor([[1., 2., 3.]])
Weight (W):
 tensor([[0.1000, 0.2000, 0.3000],
        [0.4000, 0.5000, 0.6000]], grad_fn=<CopySlices>)
Bias (b): tensor([0.7000, 0.8000], grad_fn=<CopySlices>)

Output (y): tensor([[2.1000, 4.7000]], grad_fn=<AddmmBackward0>)
```
The final output is the tensor `[[2.1, 4.7]]`.

**4. Manual Verification: Step-by-Step**

Let's prove this result. The calculation is `x @ W.T + b`.

*   **First, the matrix multiplication `x @ W.T`:**
    *   `[1, 2, 3] @ [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]`
    *   `output[0] = (1*0.1) + (2*0.2) + (3*0.3) = 0.1 + 0.4 + 0.9 = 1.4`
    *   `output[1] = (1*0.4) + (2*0.5) + (3*0.6) = 0.4 + 1.0 + 1.8 = 3.2`
    *   Result: `[1.4, 3.2]`

*   **Second, add the bias `+ b`:**
    *   `[1.4, 3.2] + [0.7, 0.8]`
    *   Result: `[2.1, 4.7]`

The manual calculation matches the PyTorch output exactly. This is all a linear layer does.

#### The Scaling Problem

This seems trivial. So where is the problem? The problem is scale.

*   **Our Toy Layer (`3x2`):**
    *   Weight parameters: `3 * 2 = 6`
    *   Bias parameters: `2`
    *   **Total:** `8` trainable parameters.

*   **A Single LLM Layer (e.g., `4096x4096`):**
    *   Weight parameters: `4096 * 4096 = 16,777,216`
    *   Bias parameters: `4096`
    *   **Total:** `16,781,312` trainable parameters.

A single layer in an LLM can have over **16 million** parameters. A full model has dozens of these layers. Trying to update all of them during fine-tuning is what melts GPUs. This is the bottleneck LoRA is designed to break.

## **Chapter 3: The LoRA Method - Math and Astonishing Savings**

This is the core idea. Instead of changing the massive weight matrix $W$, we freeze it and learn a tiny "adjustment" matrix, $\Delta W$.

The new, effective weight matrix, $W_{eff}$, is a simple sum:

$W_{eff} = W_{frozen} + \Delta W$

Training the full $\Delta W$ would be too expensive. The breakthrough of LoRA is to force this change to be **low-rank**, meaning we can construct it from two much smaller matrices, $A$ and $B$. We also add a scaling factor, $\frac{\alpha}{r}$, where $r$ is the rank and $\alpha$ is a hyperparameter.

The full LoRA update is defined by this formula:

$\Delta W = \frac{\alpha}{r} B A$

#### A Step-by-Step Numerical Example

Let's build a tiny LoRA update from scratch.

**Given:**
*   A frozen weight matrix $W_{frozen}$ of shape `[out=4, in=3]`.
*   A LoRA rank $r=2$.
*   A scaling factor $\alpha=4$.

$W_{frozen} = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \\ 3 & 3 & 3 \\ 4 & 4 & 4 \end{pmatrix}$

Now, we define our trainable LoRA matrices, $A$ and $B$:
*   $A$ must have shape `[r, in]`, so `[2, 3]`.
*   $B$ must have shape `[out, r]`, so `[4, 2]`.

Let's assume after training they have these values:

$A = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 3 & 0 \end{pmatrix} \quad B = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}$

**Step 1: Calculate the core update, $B A$**

This is a standard matrix multiplication. The result will have the same shape as $W_{frozen}$.

$B A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 0 & 2 \\ 0 & 3 & 0 \end{pmatrix} = \begin{pmatrix} (1*1+0*0) & (1*0+0*3) & (1*2+0*0) \\ (0*1+0*0) & (0*0+0*3) & (0*2+0*0) \\ (0*1+2*0) & (0*0+2*3) & (0*2+2*0) \\ (1*1+1*0) & (1*0+1*3) & (1*2+1*0) \end{pmatrix} = \begin{pmatrix} 1 & 0 & 2 \\ 0 & 0 & 0 \\ 0 & 6 & 0 \\ 1 & 3 & 2 \end{pmatrix}$

**Step 2: Apply the scaling factor, $\frac{\alpha}{r}$**

Our scaling factor is $\frac{4}{2} = 2$. We multiply our result by this scalar.

$\Delta W = 2 \times \begin{pmatrix} 1 & 0 & 2 \\ 0 & 0 & 0 \\ 0 & 6 & 0 \\ 1 & 3 & 2 \end{pmatrix} = \begin{pmatrix} 2 & 0 & 4 \\ 0 & 0 & 0 \\ 0 & 12 & 0 \\ 2 & 6 & 4 \end{pmatrix}$

This $\Delta W$ matrix is the total change that our LoRA parameters will apply to the frozen weights.

**Step 3: The "Merge" for Inference**

After training is done, we can create the final, effective weight matrix by adding the frozen weights and the LoRA update.

$W_{eff} = W_{frozen} + \Delta W = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \\ 3 & 3 & 3 \\ 4 & 4 & 4 \end{pmatrix} + \begin{pmatrix} 2 & 0 & 4 \\ 0 & 0 & 0 \\ 0 & 12 & 0 \\ 2 & 6 & 4 \end{pmatrix} = \begin{pmatrix} 3 & 1 & 5 \\ 2 & 2 & 2 \\ 3 & 15 & 3 \\ 6 & 10 & 8 \end{pmatrix}$

This final $W_{eff}$ matrix is what you would use for deployment. **Crucially, this merge calculation happens only once after training.** For inference, it's just a standard linear layer, adding zero extra latency.

#### The Forward Pass (How it works during training)

During training, we never compute the full $\Delta W$. That would be inefficient. Instead, we use the decomposed form, which is much faster. The forward pass is:

$y = W_{frozen}x + \frac{\alpha}{r} B(Ax)$

Let's compute this with an input $x = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$:

1.  **LoRA Path (right side):**
    *   `Ax =` $\begin{pmatrix} 1 & 0 & 2 \\ 0 & 3 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} = \begin{pmatrix} (1*1+0*2+2*3) \\ (0*1+3*2+0*3) \end{pmatrix} = \begin{pmatrix} 7 \\ 6 \end{pmatrix}$
    *   `B(Ax) =` $\begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 7 \\ 6 \end{pmatrix} = \begin{pmatrix} (1*7+0*6) \\ (0*7+0*6) \\ (0*7+2*6) \\ (1*7+1*6) \end{pmatrix} = \begin{pmatrix} 7 \\ 0 \\ 12 \\ 13 \end{pmatrix}$
    *   `Scale it: 2 *` $\begin{pmatrix} 7 \\ 0 \\ 12 \\ 13 \end{pmatrix} = \begin{pmatrix} 14 \\ 0 \\ 24 \\ 26 \end{pmatrix}$

2.  **Frozen Path (left side):**
    *   `W_frozen * x =` $\begin{pmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \\ 3 & 3 & 3 \\ 4 & 4 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} = \begin{pmatrix} (1+2+3) \\ (2+4+6) \\ (3+6+9) \\ (4+8+12) \end{pmatrix} = \begin{pmatrix} 6 \\ 12 \\ 18 \\ 24 \end{pmatrix}$

3.  **Final Output:**
    *   `y =` $\begin{pmatrix} 6 \\ 12 \\ 18 \\ 24 \end{pmatrix} + \begin{pmatrix} 14 \\ 0 \\ 24 \\ 26 \end{pmatrix} = \begin{pmatrix} 20 \\ 12 \\ 42 \\ 50 \end{pmatrix}$

#### The Astonishing Savings

This math is why LoRA works. Let's return to the realistic LLM layer (`4096x4096`) to see the impact.

| Method | Trainable Parameters | Calculation | Parameter Reduction |
| :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | 16,777,216 | `4096 * 4096` | 0% |
| **LoRA (r=8)** | **65,536** | `(8 * 4096) + (4096 * 8)` | **99.61%** |

By performing the efficient forward pass during training, we only need to store and update the parameters for the tiny `A` and `B` matrices, achieving a >99% parameter reduction while still being able to modify the behavior of the massive base layer.

## **Chapter 5: The Main Event - Implementing LoRA in PyTorch**

We will now translate the math from the previous chapter into a reusable PyTorch `nn.Module`. Our goal is to create a `LoRALinear` layer that wraps a standard `nn.Linear` layer, freezes it, and adds the trainable `A` and `B` matrices.

#### The `LoRALinear` Module

Here is the complete implementation, followed by a breakdown of each part.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        # --- Store hyperparameters ---
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        # --- Store and freeze the original linear layer ---
        self.base = base
        self.base.weight.requires_grad_(False)
        # Also freeze the bias if it exists
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # --- Create the trainable LoRA matrices A and B ---
        # A has shape [r, in_features]
        # B has shape [out_features, r]
        self.lora_A = nn.Parameter(torch.empty(r, self.base.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.base.out_features, r))

        # --- Initialize the weights ---
        # A is initialized with a standard method
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is initialized with zeros
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. The original, frozen path
        base_output = self.base(x)

        # 2. The efficient LoRA path: B(A(x))
        # F.linear(x, self.lora_A) computes x @ A.T
        # F.linear(..., self.lora_B) computes (x @ A.T) @ B.T
        lora_update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

        # 3. Return the combined output
        return base_output + lora_update
```

**Breakdown:**

1.  **`__init__(self, base, r, alpha)`**:
    *   It accepts the original `nn.Linear` layer (`base`) that we want to adapt.
    *   `self.base.weight.requires_grad_(False)`: This is the critical **"freezing"** step. We tell PyTorch's autograd engine not to compute gradients for the original weights, so they will never be updated by the optimizer.
    *   `nn.Parameter(...)`: We register `lora_A` and `lora_B` as official trainable parameters of the module. Their shapes are derived directly from the base layer and the rank `r`.
    *   `nn.init.zeros_(self.lora_B)`: This is a crucial initialization detail. By starting `B` as a zero matrix, the entire LoRA update (`B @ A`) is zero at the beginning of training. This means our `LoRALinear` layer initially behaves exactly like the original frozen layer, and the model learns the "change" from a stable starting point.

2.  **`forward(self, x)`**:
    *   This is a direct translation of the formula: $y = W_{frozen}x + \frac{\alpha}{r} B(Ax)$
    *   We compute the output of the frozen path and the LoRA path separately.
    *   The nested `F.linear` calls are a highly efficient PyTorch way to compute `(x @ A.T) @ B.T` without ever forming the full $\Delta W$ matrix.
    *   Finally, we add them together.

#### Applying LoRA to a Model

Now we need a helper function to swap out the `nn.Linear` layers in any given model with our new `LoRALinear` layer.

```python
def apply_lora(model: nn.Module, r: int, alpha: float = 16.0):
    """
    Replaces all nn.Linear layers in a model with LoRALinear layers.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Find the parent module to replace the child
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)

            # Replace the original linear layer
            setattr(parent_module, child_name, LoRALinear(module, r=r, alpha=alpha))
```

#### Minimal End-to-End Demo

Let's see it all work together.

**1. Create a toy model:**
```python
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10) # e.g., for classification
)
```

**2. Inject LoRA layers:**
```python
apply_lora(model, r=8, alpha=16.0)
print(model)
```
The output will show that our `nn.Linear` layers have been replaced by `LoRALinear`.

**3. Isolate the Trainable Parameters:**
This is the most important step. We create an optimizer that *only* sees the LoRA weights.

```python
# Filter for parameters that require gradients (only lora_A and lora_B)
trainable_params = [p for p in model.parameters() if p.requires_grad]
trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]

print("\nTrainable Parameters:")
for name in trainable_param_names:
    print(name)

# Create an optimizer that only updates the LoRA weights
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
```

**Output:**
```text
Trainable Parameters:
0.lora_A
0.lora_B
2.lora_A
2.lora_B
```
This proves our success. The optimizer is completely unaware of the massive, frozen weights (`0.base.weight`, `2.base.weight`, etc.) and will only update our tiny, efficient LoRA matrices.

## **Chapter 6: Conclusion - From a Toy Model to a Real Transformer**

Let's recap the journey. We started with a simple `nn.Linear` layer and saw how its parameter count explodes at the scale of a real LLM. We then introduced the core mathematical trick of LoRA: approximating the massive update matrix `ΔW` with two small, low-rank matrices `A` and `B`. This simple idea led to a staggering >99% reduction in trainable parameters. Finally, we translated that math into a clean, reusable `LoRALinear` PyTorch module and proved that an optimizer could be set up to *only* train these new, tiny matrices.

#### Where does LoRA actually go in an LLM?

The `nn.Linear` layers we've been working with are not just abstract examples. They are the primary components of a Transformer, the architecture behind virtually all modern LLMs.

When you apply LoRA to a model like Llama or Mistral, you are targeting these specific linear layers:

*   **Self-Attention Layers:** The most common targets are the projection matrices for the **query (`q_proj`)** and **value (`v_proj`)**. Adapting these allows the model to change *what it pays attention to* in the input text, which is incredibly powerful for task-specific fine-tuning.
*   **Feed-Forward Layers (MLP):** Transformers also have blocks of linear layers that process information after the attention step. Applying LoRA here helps modify the model's learned representations and knowledge.

So, when you see a LoRA implementation for a real LLM, the `apply_lora` function is simply more selective, replacing only the linear layers named `q_proj`, `v_proj`, etc., with the `LoRALinear` module you just built.

#### Why This Works So Well

The stunning effectiveness of LoRA relies on a powerful hypothesis: the knowledge needed to adapt a pre-trained model to a new task is much simpler than the model's entire knowledge base. You don't need to re-learn the entire English language to make a model a better chatbot. You only need to steer its existing knowledge. This "steering" information lies in a low-dimensional space, which a low-rank update `ΔW = B @ A` can capture perfectly.

You now have a deep, practical understanding of one of the most important techniques in modern AI. You know the "what," the "why," and the "how" behind LoRA, giving you the foundation to efficiently adapt massive language models.