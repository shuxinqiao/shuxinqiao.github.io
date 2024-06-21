+++
title = "Fourier Neural Operator Expansion"
+++

## Related Links

- [Author's Github.io](https://zongyi-li.github.io/) \
- [Orignial Paper Arxiv](https://arxiv.org/abs/2010.08895)

## 4D Expansion (Time)

Since using **Fourier Transform** (*Fast Fourier Transform (FFT)* in implementation level), FNO is naturally able to deal with both space-domain and time-domain information.

By simply adding one more dimensions into **Einsum** (Einstein Summation Convention), it expands to a fixed Time-Space 4D structure.

```python
# 3D with x,y,z
torch.einsum("bixyz,ioxyz->boxyz", input, weights)

# into 4D with x,y,z,t
torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)
```

&nbsp;

Noticably, the weight parameter amount is *multiplication* relation to both **modes** and **dimensions**.


Here is an example:

| Modes         | Dimensions    | Parameter Amount    |
| ------------- |:-------------:| -------------------:|
| [12,30,30]    | 3             |   12x30x30 = 10,800 |
| [5,12,30,30]  | 4             | 5x12x30x30 = 54,000 |
| [5,6,15,15]   | 4             |   5x6x15x15 = 6,750 |

&nbsp;

FNO's ability is hidding inside **extracted wavelets**, it makes modes the most important hyperparameter to a FNO model.
Other hyperparameters like *number of layers*, *whether use residual connections* and etc are incompetitive compare to modes.

Finding balance in a sufficient and efficient number of modes with available computation resources is a great question.

If modes appeared to be more important in the task or variant time steps is desired, please consider RNN, LSTM, Structured State Space or other time-series structure.