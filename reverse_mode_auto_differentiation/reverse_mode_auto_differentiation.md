# Reverse-Mode Automatic differentiation
---
Some references (only list them here first):

- [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)
- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/gradients.py#L308)
- [the question/answer in Corss Validated](https://stats.stackexchange.com/questions/224140/step-by-step-example-of-reverse-mode-automatic-differentiation)
- a post
    - [Automatic Differentiation: The most criminally underused tool in the potential machine learning toolbox?](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/) 
    - [A Simple Explanation of Reverse Mode Automatic Differentiation](https://justindomke.wordpress.com/2009/03/24/a-simple-explanation-of-reverse-mode-automatic-differentiation/)
- a web site: http://www.autodiff.org/

---
# Revisit to Backpropagation
---
- A neural network can be described as a series of altering **linear transformations** and **elementwise nonlinearities**.
$$\begin{align}\begin{split}S_n&= W_n \mathbf{x}_{n-1} \\ \mathbf{x}_n &= \sigma(S_n) \end{split}\end{align}$$

- $\sigma$ is a sigmoid function（nonlinearity）
- present input ${\mathbf x}_0$ and get output ${\mathbf x}_N$.
- some loss function $L$ that says how much you like that particular ouput on that input.

--- 
- Backpropagation is an algorithm for calculating the derivatives of $L$ with respect to all the weight matrices $W_n$.
- Backpropagation is a special case of reverse-mode automatic differentiation.

---
- The loss function will directly give the derivatives with respect to the output, this will be either $\frac{dL}{d{\mathbf x}_N}$ or $\frac{dL}{d{\mathbf s}_N}$.
- it is very easy to conclude the following three rules:

$$ \begin{align}\begin{split} \frac{dL}{d{\mathbf s}_n} &= \frac{dL}{d{\mathbf x}_n} \odot \sigma'({\mathbf s}_n) \\
\frac{dL}{dW_n} &= \frac{dL}{d{\mathbf s}_n}{{\mathbf x}_{n-1}}^T \\
\frac{dL}{d{\mathbf x}_{n-1}} &= {W_n}^T\frac{dL}{d{\mathbf s}_n} \end{split}\end{align}$$

- Notice that **applying these rules in reverse will have the same complexity as the original “forward propagation”**.
---

# automatic differentiation?
- to compute the gradient, autograd first has to **record every transformation that was applied to the input** as it was turned into the output of your function.
- autograd wraps functions so that when they're called, they add themselves to a list of operations performed.
- After the function is evaluated, autograd has a list of all operations that were performed and which nodes they depended on.
- This is the computational graph of the function evaluation.
- To compute the derivative, we simply apply the rules of differentiation to each node in the graph.
---

# What is reverse model auto differentiation?
