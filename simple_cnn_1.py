import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Exact Architecture
class MinimalNetwork(nn.Module):
    def __init__(self):
        super(MinimalNetwork, self).__init__()
        # 2 inputs -> 2 hidden neurons
        self.hidden = nn.Linear(2, 2)
        # 2 hidden neurons -> 2 output neurons
        self.output = nn.Linear(2, 2)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights and biases to match the math exactly
        with torch.no_grad():
            # Input-to-Hidden weights (PyTorch uses transposed shapes: [out_features, in_features])
            self.hidden.weight.copy_(torch.tensor([[0.15, 0.20],  # w1, w2
                                                   [0.25, 0.30]])) # w3, w4
            self.hidden.bias.copy_(torch.tensor([0.35, 0.35]))     # b1
            
            # Hidden-to-Output weights
            self.output.weight.copy_(torch.tensor([[0.40, 0.45],  # w5, w6
                                                   [0.50, 0.55]])) # w7, w8
            self.output.bias.copy_(torch.tensor([0.60, 0.60]))     # b2

    def forward(self, x):
        h = self.sigmoid(self.hidden(x))
        o = self.sigmoid(self.output(h))
        return h, o

def custom_squared_error(outputs, targets):
    return 0.5 * torch.mean((targets - outputs) ** 2)


def main():
    # 2. Setup Data and Hyperparameters
    batch_size = 10
    inputs = torch.tensor([[0.05, 0.10]], dtype=torch.float32).repeat(batch_size, 1)
    targets = torch.tensor([[0.01, 0.99]], dtype=torch.float32).repeat(batch_size, 1)

    model = MinimalNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.5)

    print("--- Initial Weights ---")
    print(f"  Input-to-Hidden:  w1={model.hidden.weight[0, 0].item():+.2f}  w2={model.hidden.weight[0, 1].item():+.2f}  w3={model.hidden.weight[1, 0].item():+.2f}  w4={model.hidden.weight[1, 1].item():+.2f}")
    print(f"  Hidden-to-Output: w5={model.output.weight[0, 0].item():+.2f}  w6={model.output.weight[0, 1].item():+.2f}  w7={model.output.weight[1, 0].item():+.2f}  w8={model.output.weight[1, 1].item():+.2f}\n")

    epochs = 100000
    for epoch in range(1, epochs + 1):
        # Forward Pass
        hidden_out, final_out = model(inputs)
        loss = custom_squared_error(final_out, targets)

        if epoch == 1:
            print("--- First Batch Output ---")
            print(f"  Hidden layer outputs (all {batch_size} samples):\n{hidden_out.detach().numpy()}")
            print(f"  Final predictions   (all {batch_size} samples):\n{final_out.detach().numpy()}")
            print(f"  Loss: {loss.item():.6f}\n")

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1:
            print("--- Updated weights after epoch 1 ---")
            print(f"  w1: {model.hidden.weight[0, 0].item():+.12f}  w2: {model.hidden.weight[0, 1].item():+.12f}")
            print(f"  w3: {model.hidden.weight[1, 0].item():+.12f}  w4: {model.hidden.weight[1, 1].item():+.12f}")
            print(f"  w5: {model.output.weight[0, 0].item():+.12f}  w6: {model.output.weight[0, 1].item():+.12f}")
            print(f"  w7: {model.output.weight[1, 0].item():+.12f}  w8: {model.output.weight[1, 1].item():+.12f}\n")

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | "
                  f"out_o1: {final_out[0, 0].item():.6f} | out_o2: {final_out[0, 1].item():.6f}")

    print(f"\n--- Final parameters after {epochs} epochs ---")
    print(f"  w1: {model.hidden.weight[0, 0].item():+.12f}  w2: {model.hidden.weight[0, 1].item():+.12f}")
    print(f"  w3: {model.hidden.weight[1, 0].item():+.12f}  w4: {model.hidden.weight[1, 1].item():+.12f}")
    print(f"  w5: {model.output.weight[0, 0].item():+.12f}  w6: {model.output.weight[0, 1].item():+.12f}")
    print(f"  w7: {model.output.weight[1, 0].item():+.12f}  w8: {model.output.weight[1, 1].item():+.12f}\n")

    # Inference
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[0.05, 0.10]], dtype=torch.float32)
        hidden_out, final_out = model(test_input)
    print("--- Inference ---")
    print(f"  Input:          {test_input.numpy()[0]}")
    print(f"  Target:         [0.01, 0.99]")
    print(f"  Hidden output:  {hidden_out.numpy()[0]}")
    print(f"  Final output:   {final_out.numpy()[0]}")
    print(f"  Predicted:      o1={final_out[0, 0].item():+.12f}  o2={final_out[0, 1].item():+.12f}")


if __name__ == "__main__":
    main()