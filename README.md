# ASD(FGHJK)L
The library is called "ASDL", which stands for **A**utomatic **S**econd-order **D**ifferentiation (for **F**isher, **G**radient covariance, **H**essian, **J**acobian, and **K**ernel) **L**ibrary.

You can import `asdfghjkl` by scrolling your finger on a QWERTY keyboard :innocent:
```python
import asdfghjkl
```

## Basic metrics supported by an automatic differentiation libarary (ADL)
| metric | definition |
| --- | --- |
| neural network | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;f_\theta:\mathbb{R}^{M_{in}}\to\mathbb{R}^{C},\,\,\,\theta\in\mathbb{R}^{P}"/> |
| loss | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^N\ell(x_i,y_i,\theta)=\left\langle\ell(x_i,y_i,\theta)\right\rangle"/> |
| gradient | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\bar{g}=\nabla\mathcal{L}(\theta)=\left\langle\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)\right\rangle=\left\langle\mathbf{J}_{f,\theta}(x_i)^\top\frac{\partial}{\partial{f}}\ell(x_i,y_i,\theta)\right\rangle\in\mathbb{R}^P"/> |

## Advanced metrics (FGHJK) supported by ASDL
| metric | definition |
| --- | --- |
| **F**isher information matrix | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\mathbf{F}=\left\langle\mathbb{E}_{p(k\|x_i)}\left[\frac{\partial}{\partial\theta}\ell(x_i,k,\theta)\frac{\partial}{\partial\theta}\ell(x_i,k,\theta)^\top\right]\right\rangle\in\mathbb{R}^{P\times{P}}" />  |
| **G**radient covariance | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\mathbf{C}=\left\langle\left(\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)-\bar{g}\right)\left(\frac{\partial}{\partial\theta}\ell(x_i,y_i,\theta)-\bar{g}\right)^\top\right\rangle\in\mathbb{R}^{P\times{P}}" />  |
| **H**essian | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\mathbf{H}=\left\langle\frac{\partial^2}{\partial\theta\theta^\top}\ell(x_i,y_i,\theta)\right\rangle\in\mathbb{R}^{P\times{P}}"/> |
| **J**acobian (per sample) | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\mathbf{J}_{f,\theta}(x)=\frac{\partial}{\partial\theta}f_{\theta}(x)\in\mathbb{R}^{C\times{P}}"/> |
| **J**acobian | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\mathcal{J}=\left[\mathbf{J}_{f,\theta}(x_1)^\top,\dots,\mathbf{J}_{f,\theta}(x_N)^\top\right]^\top\in\mathbb{R}^{NC\times{P}}"/> |
| **K**ernel | <img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\mathcal{K}=\mathcal{JJ}^\top\in\mathbb{R}^{NC\times{NC}}"/> |
