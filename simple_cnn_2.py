import torch
import torch.nn as nn
import torch.optim as optim

# ── Architecture ──────────────────────────────────────────────────────────────
# 2 inputs → 2 hidden (sigmoid) → 2 outputs (sigmoid)
# Weights and biases are set manually so results match hand-calculated values.
class MinimalNetwork(nn.Module):
    def __init__(self, w_hidden=None, w_output=None):
        super(MinimalNetwork, self).__init__()
        self.hidden = nn.Linear(2, 2)   # w1-w4, bias b1
        self.output = nn.Linear(2, 2)   # w5-w8, bias b2
        self.sigmoid = nn.Sigmoid()

        # Override random init with caller-supplied (or textbook default) values
        with torch.no_grad():
            h = w_hidden if w_hidden is not None else [[0.15, 0.20], [0.25, 0.30]]
            o = w_output if w_output is not None else [[0.40, 0.45], [0.50, 0.55]]
            self.hidden.weight.copy_(torch.tensor(h))
            self.hidden.bias.copy_(torch.tensor([0.35, 0.35]))
            self.output.weight.copy_(torch.tensor(o))
            self.output.bias.copy_(torch.tensor([0.60, 0.60]))

    def forward(self, x):
        h = self.sigmoid(self.hidden(x))    # hidden activations
        o = self.sigmoid(self.output(h))    # output activations
        return h, o

# ── Loss ──────────────────────────────────────────────────────────────────────
# Standard squared-error scaled by 0.5 so the gradient simplifies to (pred - target).
def custom_squared_error(outputs, targets):
    return 0.5 * torch.mean((targets - outputs) ** 2)

# ── Weight printer ────────────────────────────────────────────────────────────
# Prints w1-w8 in four lines (one pair per line).
# fmt:    Python format spec for each value  (default: 12 decimal places)
# indent: leading whitespace before each line
def print_weights(model, fmt="+.12f", indent="  "):
    w, v = model.hidden.weight, model.output.weight
    print(f"{indent}w1: {w[0,0].item():{fmt}}  w2: {w[0,1].item():{fmt}}")
    print(f"{indent}w3: {w[1,0].item():{fmt}}  w4: {w[1,1].item():{fmt}}")
    print(f"{indent}w5: {v[0,0].item():{fmt}}  w6: {v[0,1].item():{fmt}}")
    print(f"{indent}w7: {v[1,0].item():{fmt}}  w8: {v[1,1].item():{fmt}}")

# ── Training run ──────────────────────────────────────────────────────────────
# Trains one model from scratch with the given initial weights, then runs
# a single inference to check how close the outputs are to the targets.
def train_and_infer(label, w_hidden, w_output, batch_size=10, epochs=100000):
    # Repeat the single sample to form a batch (all rows are identical)
    inputs  = torch.tensor([[0.05, 0.10]], dtype=torch.float32).repeat(batch_size, 1)
    targets = torch.tensor([[0.01, 0.99]], dtype=torch.float32).repeat(batch_size, 1)

    model = MinimalNetwork(w_hidden=w_hidden, w_output=w_output)
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    print(f"\n{'='*60}")
    print(f"  Run: {label}")
    print(f"{'='*60}")
    print_weights(model, fmt="+.4f")    # show starting weights at low precision
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    for _ in range(1, epochs + 1):
        _, final_out = model(inputs)
        loss = custom_squared_error(final_out, targets)
        optimizer.zero_grad()   # clear gradients from previous step
        loss.backward()         # compute gradients
        optimizer.step()        # update weights

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"  Final parameters after {epochs} epochs:")
    print_weights(model, indent="    ")     # full precision for comparison

    # Single-sample inference (no gradient needed)
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[0.05, 0.10]], dtype=torch.float32)
        _, final_out = model(test_input)
    print(f"\n  Inference → o1={final_out[0,0].item():+.12f}  o2={final_out[0,1].item():+.12f}")
    print(f"  Target    → o1=+0.010000000000  o2=+0.990000000000")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # Run A: textbook starting weights — the canonical worked example
    train_and_infer(
        label="Init A (textbook: w1=0.15 ... w8=0.55)",
        w_hidden=[[0.15, 0.20], [0.25, 0.30]],
        w_output=[[0.40, 0.45], [0.50, 0.55]],
    )

    # Run B: different starting weights — shows the network still converges
    train_and_infer(
        label="Init B (different: w1=0.80 ... w8=0.20)",
        w_hidden=[[0.80, 0.10], [0.60, 0.40]],
        w_output=[[0.20, 0.70], [0.30, 0.90]],
    )


if __name__ == "__main__":
    main()
