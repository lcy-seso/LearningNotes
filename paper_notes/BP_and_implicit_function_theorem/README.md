[TOC]

# Backpropagation and implicit function theorem

$$f(x) = \text{exp}(\text{exp}(x) + \text{exp}(x)^2) + \text{sin}(\text{exp}(x) + \text{exp}(x)^2)$$

```python
def f(x)
    a = exp(x)
    b = a ** 2
    c = a + b
    d = exp(c)
    e = sin(c)
    return d + e
```

## Reference

1. [Backprop is not just the chain rule](https://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/)
1. [A new trick for calculating Jacobian vector products](https://j-towns.github.io/2017/06/12/A-new-trick.html)
1. [Mechanics of Lagrangians](http://www.argmin.net/2016/05/31/mechanics-of-lagrangians/)
1. [Mates of Costate](http://www.argmin.net/2016/05/18/mates-of-costate/)
