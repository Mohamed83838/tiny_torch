# AutoGrad Matrix Multiplication Example

## Overview

This example demonstrates a forward pass and backpropagation using a custom `tiny_torch` library with `Tensor` and `AutoGrad` support.

## Formula

```
n = (ll * (x @ y)) + (l + (x @ y))
```

## Code

```cpp
#include <iostream>
#include "Tensor.h"
#include "AutoGrad.h"

int main() {
    // Define matrix dimensions
    std::vector<size_t> shape1 = { 4, 8 };
    std::vector<size_t> shape2 = { 8, 9 };
    std::vector<size_t> shape3 = { 4, 9 };

    // Initialize Tensors
    Tensor x = tiny_torch::fill(shape1, 55.0f);
    Tensor y = tiny_torch::rand(shape2);
    Tensor l = tiny_torch::rand(shape3);
    Tensor ll = tiny_torch::rand(shape3);

    // Enable gradient tracking
    x.auto_grad = true;
    y.auto_grad = true;

    // Forward Pass: n = (ll * (x @ y)) + (l + (x @ y))
    Tensor z = tiny_torch::mm(&x, &y);
    Tensor t = tiny_torch::add(&l, &z);
    Tensor k = tiny_torch::multiply(&ll, &z);
    Tensor n = tiny_torch::add(&k, &t);

    // Run Backpropagation
    tiny_torch::auto_grad::backward(n.grad_fn);

    // Access Gradients
    std::cout << "Gradient of x:\n" << x.grad_tensor()->print() << std::endl;
    std::cout << "Gradient of y:\n" << y.grad_tensor()->print() << std::endl;

    return 0;
}
```

## Tensor Shapes

| Tensor | Shape  | Initialization |
|--------|--------|----------------|
| `x`    | 4 × 8  | Filled with `55.0f` |
| `y`    | 8 × 9  | Random         |
| `l`    | 4 × 9  | Random         |
| `ll`   | 4 × 9  | Random         |

## Computation Graph

| Step | Operation              | Result |
|------|------------------------|--------|
| 1    | `z = x @ y`            | 4 × 9  |
| 2    | `t = l + z`            | 4 × 9  |
| 3    | `k = ll * z`           | 4 × 9  |
| 4    | `n = k + t`            | 4 × 9  |

## Notes

- Only `x` and `y` have gradient tracking enabled (`auto_grad = true`).
- `l` and `ll` are treated as constants with no gradient computation.
- Backpropagation is triggered via `tiny_torch::auto_grad::backward(n.grad_fn)`.
