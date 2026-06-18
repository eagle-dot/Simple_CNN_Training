import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Exact Architecture
class MinimalNetwork(nn.Module):
    def __init__(self, w_hidden=None, w_output=None):
        super(MinimalNetwork, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 2)
        self.sigmoid = nn.Sigmoid()

        with torch.no_grad():
            h = w_hidden if w_hidden is not None else [[0.15, 0.20], [0.25, 0.30]]
            o = w_output if w_output is not None else [[0.40, 0.45], [0.50, 0.55]]
            self.hidden.weight.copy_(torch.tensor(h))
            self.hidden.bias.copy_(torch.tensor([0.35, 0.35]))
            self.output.weight.copy_(torch.tensor(o))
            self.output.bias.copy_(torch.tensor([0.60, 0.60]))

    def forward(self, x):
        h = self.sigmoid(self.hidden(x))
        o = self.sigmoid(self.output(h))
        return h, o

def custom_squared_error(outputs, targets):
    return 0.5 * torch.mean((targets - outputs) ** 2)


def train_and_infer(label, w_hidden, w_output, batch_size=10, epochs=100000):
    inputs  = torch.tensor([[0.05, 0.10]], dtype=torch.float32).repeat(batch_size, 1)
    targets = torch.tensor([[0.01, 0.99]], dtype=torch.float32).repeat(batch_size, 1)

    model = MinimalNetwork(w_hidden=w_hidden, w_output=w_output)
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    print(f"\n{'='*60}")
    print(f"  Run: {label}")
    print(f"{'='*60}")
    print(f"  Initial  w1={model.hidden.weight[0,0].item():+.4f}  w2={model.hidden.weight[0,1].item():+.4f}  "
          f"w3={model.hidden.weight[1,0].item():+.4f}  w4={model.hidden.weight[1,1].item():+.4f}")
    print(f"  Initial  w5={model.output.weight[0,0].item():+.4f}  w6={model.output.weight[0,1].item():+.4f}  "
          f"w7={model.output.weight[1,0].item():+.4f}  w8={model.output.weight[1,1].item():+.4f}\n")

    for _ in range(1, epochs + 1):
        _, final_out = model(inputs)
        loss = custom_squared_error(final_out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  Final parameters after {epochs} epochs:")
    print(f"    w1: {model.hidden.weight[0,0].item():+.12f}  w2: {model.hidden.weight[0,1].item():+.12f}")
    print(f"    w3: {model.hidden.weight[1,0].item():+.12f}  w4: {model.hidden.weight[1,1].item():+.12f}")
    print(f"    w5: {model.output.weight[0,0].item():+.12f}  w6: {model.output.weight[0,1].item():+.12f}")
    print(f"    w7: {model.output.weight[1,0].item():+.12f}  w8: {model.output.weight[1,1].item():+.12f}")

    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[0.05, 0.10]], dtype=torch.float32)
        _, final_out = model(test_input)
    print(f"\n  Inference → o1={final_out[0,0].item():+.12f}  o2={final_out[0,1].item():+.12f}")
    print(f"  Target    → o1=+0.010000000000  o2=+0.990000000000")


def main():
    # Initialization A — original textbook values
    train_and_infer(
        label="Init A (textbook: w1=0.15 ... w8=0.55)",
        w_hidden=[[0.15, 0.20], [0.25, 0.30]],
        w_output=[[0.40, 0.45], [0.50, 0.55]],
    )

    # Initialization B — different starting weights
    train_and_infer(
        label="Init B (different: w1=0.80 ... w8=0.20)",
        w_hidden=[[0.80, 0.10], [0.60, 0.40]],
        w_output=[[0.20, 0.70], [0.30, 0.90]],
    )


if __name__ == "__main__":
    main()