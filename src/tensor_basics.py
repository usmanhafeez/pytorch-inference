
import torch

# 1. Create a 1D tensor
x = torch.tensor([1.0, 2.0, 3.0])
print("1D tensor:", x)
print("Shape:", x.shape)

# 2. Create a 2D tensor (matrix)
matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
print("\n2D tensor:")
print(matrix)
print("Shape:", matrix.shape)

# 3. Element-wise multiplication
result = matrix * 2
print("\nElement-wise multiplication:")
print(result)

# 4. Matrix multiplication
a = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

b = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

c = torch.matmul(a, b)
print("\nMatrix multiplication:")
print(c)

# 5. CPU information
print("\nDevice:", x.device)
print("CPU threads:", torch.get_num_threads())

