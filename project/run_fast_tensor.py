import random

import numba

import minitorch

from minitorch.cuda_ops import CudaOps
from minitorch.tensor import Tensor
from minitorch.tensor_data import TensorData
from minitorch.tensor_functions import TensorBackend
from minitorch.fast_ops import FastOps

# Add this line to import GPUBackend
from minitorch.cuda_ops import CudaOps as GPUBackend

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape, backend):
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        super().__init__()
        # Much wider network
        self.layer1 = Linear(2, hidden * 8, backend)
        self.layer2 = Linear(hidden * 8, hidden * 4, backend)
        self.layer3 = Linear(hidden * 4, 1, backend)

    def forward(self, x):
        # Less aggressive regularization
        h = self.layer1.forward(x).relu() * 0.9
        h = self.layer2.forward(h).relu() * 0.9
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        # Better weight initialization
        self.weights = RParam(in_size, out_size, backend=backend)
        self.weights.value = self.weights.value * 0.05  # Smaller initial weights
        
        # Initialize bias with zeros
        self.bias = minitorch.Parameter(minitorch.zeros((out_size,), backend=backend))
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        # Add small epsilon for numerical stability
        out = x @ self.weights.value
        return out + self.bias.value.view(1, self.out_size)


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.model = Network(self.hidden_layers, self.backend)
        # Adjust learning rate
        optim = minitorch.SGD(self.model.parameters(), learning_rate * 0.01)
        BATCH = 1  # Single sample batch
        patience = 100  # Much more patience
        min_delta = 0.00001  # Smaller improvement threshold
        
        # Initialize early stopping variables inside train method
        best_loss = float('inf')
        patience_counter = 0
        
        # Create tensor constants
        one = minitorch.tensor([1.0], backend=self.backend)
        eps = minitorch.tensor([1e-8], backend=self.backend)

        losses = []
        for epoch in range(max_epochs):
            total_loss = 0.0
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)

            batch_losses = []
            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()
                X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)
                
                # Forward pass
                out = self.model.forward(X).view(y.shape[0])
                
                # Compute stable loss using tensors
                out = out * 0.99 + 0.005
                y_term = y * (out + eps).log()
                not_y_term = (one - y) * (one - out + eps).log()
                loss = -(y_term + not_y_term).sum()
                
                batch_losses.append(loss.detach()[0])
                loss.backward()
                optim.step()

            total_loss = sum(batch_losses) / len(batch_losses)
            losses.append(total_loss)
            
            # Early stopping check
            if total_loss < best_loss - min_delta:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 5 == 0 or epoch == max_epochs - 1:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--PLOT", default=False, help="dataset")
    parser.add_argument("--MAX_EPOCHS", type=int, default=50, help="maximum number of epochs")

    args = parser.parse_args()

    PTS = args.PTS

    # Fix dataset access - use dictionary access with correct keys
    if args.DATASET == "xor":
        data = datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = datasets["Simple"](PTS)
    elif args.DATASET == "split":
        data = datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    # Create and train model
    model = FastTrain(
        HIDDEN, 
        backend=FastTensorBackend if args.BACKEND != "gpu" else GPUBackend
    )
    
    # Train with early stopping
    model.train(data, RATE, max_epochs=args.MAX_EPOCHS)
