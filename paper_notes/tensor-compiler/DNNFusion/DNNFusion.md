
# [DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://arxiv.org/pdf/2108.13342.pdf)

## Approach

It is possible to classify all operators into **five** high-level abstract types based on the relationship between input and output elements.

>***<font color=#5F9EA0>Though in DNNFusion, the author does not give a formal definition of the concept of "mapping type." In my eye, it describes the information that the access function captures in polyhedral compilation.</font>.***

Assume each input element can be denoted as $x[d_1, ..., d_n]$:

>1. <font color=#5F9EA0>Tensor a is collection type.</font>
>1. <font color=#5F9EA0>fusion analysis in paper mainly keeps operators black-box. The mapping type is still very vague. It is not carefully distinguish (the analysis is not sensitive to ?) whether $d_{...}$ an integer(one-dimensional addressing) or tuple (high-dimensional addressing)?</font>

1. **One-to-One**: $y[d_1,...,d_n]=F(x[f_1(d_1),...,f_n(d_n)])$

   >***It seems for me that $f_1$...$f_n$ are just a same function, they are not different. One-to-one***
    - shape does not change.

2. **Reorganize**

    > this class for me it to change the intepretation of logical address (modifying meta information).

3. **Shuffle**: $y[e_1,...,e_n]=x[f_1(d_{F(1)}),...,f_n(f_{F(n)})]$

    a set permutation function.

4. **One-to-Many**: $y[e_1,...,e_m]=F(x[f_1(d_1),...,f_n(d_n)])$, $m>n$

5. <font color=#B22222>**Many-to-Many**</font>:  $y[e_1,...,e_m]=F(x^1[f_1^1(d_1), ..., f_n^1(d_n)],...,x^k[f_1^k(d_1), ..., f_k^k(d_n)])$

    <font color=red>This definition DOES not seem correct for me</font>.

<p align="center">
<img src="images/mapping_types.png"><br>
Highlighted are some operators that the classification does not appropriately capture the key information in optimizing its implementation, in my opinion.
</p>

<p align="center">
<img src="images/fusion.png" width=80%><br>
