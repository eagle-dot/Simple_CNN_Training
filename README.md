# Neural Network Backpropagation Training Walkthrough

A minimal PyTorch implementation of a fully-connected neural network (MLP) that walks through forward pass, backpropagation, and weight updates step by step — built to verify hand-calculated math from a classic backprop example.

---

## Network Architecture

A **2-layer MLP** (2 inputs → 2 hidden neurons → 2 outputs) with sigmoid activations.

| Component | Value |
|---|---|
| Inputs | x₁ = 0.05, x₂ = 0.10 |
| Hidden layer | 2 neurons, bias b₁ = 0.35 |
| Output layer | 2 neurons, bias b₂ = 0.60 |
| Targets | t₁ = 0.01, t₂ = 0.99 |
| Activation | Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ) |
| Learning rate | η = 0.5 |
| Optimizer | SGD |

**Initial weights:**

- Input-to-Hidden: w₁ = 0.15, w₂ = 0.20, w₃ = 0.25, w₄ = 0.30
- Hidden-to-Output: w₅ = 0.40, w₆ = 0.45, w₇ = 0.50, w₈ = 0.55

---

## Forward Pass (Epoch 1)

**Hidden layer:**
```
net_h1 = (0.15)(0.05) + (0.20)(0.10) + 0.35 = 0.3775  →  out_h1 = σ(0.3775) = 0.593265
net_h2 = (0.25)(0.05) + (0.30)(0.10) + 0.35 = 0.3925  →  out_h2 = σ(0.3925) = 0.596884
```

**Output layer:**
```
net_o1 = (0.40)(0.593265) + (0.45)(0.596884) + 0.60 = 1.105904  →  out_o1 = 0.751365
net_o2 = (0.50)(0.593265) + (0.55)(0.596884) + 0.60 = 1.224919  →  out_o2 = 0.772928
```

**Total error (squared error):**
```
E_o1     = ½(0.01 − 0.751365)² = 0.274811
E_o2     = ½(0.99 − 0.772928)² = 0.023560
E_total  = 0.298371
```

---

## Backward Pass (Backpropagation)

### Phase 1 — Updating Hidden-to-Output weights (w₅ example)

Chain rule:

```
∂E_total/∂w₅ = (∂E_total/∂out_o1) × (∂out_o1/∂net_o1) × (∂net_o1/∂w₅)
             = 0.741365 × 0.186816 × 0.593265
             = 0.082167

w₅_new = 0.40 − (0.5)(0.082167) = 0.358916
```

### Phase 2 — Updating Input-to-Hidden weights (w₁ example)

Pool downstream errors first:

```
∂E_total/∂out_h1 = [(∂E_o1/∂net_o1) × w₅] + [(∂E_o2/∂net_o2) × w₇]
                 = 0.055399 + (−0.019030) = 0.036369

∂E_total/∂w₁ = 0.036369 × 0.241301 × 0.05 = 0.0004388

w₁_new = 0.15 − (0.5)(0.0004388) = 0.149780
```

---

## Weight Summary After 1 Epoch

| Weight | Initial | After Epoch 1 |
|---|---|---|
| w₁ (Input → Hidden) | 0.15 | 0.149780 |
| w₅ (Hidden → Output) | 0.40 | 0.358916 |

---

## Usage

```bash
pip install torch
python simple_cnn_1.py
```

The script prints:
- Initial weights
- First-epoch hidden and output activations
- Updated weights after epoch 1
- Loss at every 10th epoch
- Final weights and inference result after full training

---

## Notes
Clck the following for a better view
https://eagle-dot.github.io/Simple_CNN_Training/cnn.html

- This is a **fully connected MLP**, not a CNN despite the filename. Converting to a true CNN would require convolutional layers (spatial kernels) and pooling layers before the fully connected output.
- The loss function uses `0.5 × mean((target − output)²)` so gradient magnitude stays consistent regardless of batch size.
- Different weight initializations converge to different final weight matrices, but all produce predictions close to `[0.01, 0.99]` — demonstrating that the loss surface has multiple valid solutions.
