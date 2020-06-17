# Tensor shape

Tensor shape $S=<S_1,S_2,...,S_n>$ is a tuple of n non-negative integers that specify the sizes of each dimension of a Tensor.

- A tuple is a collection of objects which ordered and immutable.
- A tuple is written as two parentheses containing integers in it.

    ```python
    (3,5,3)
    () # empty tuple
    ```

## Shape Operations

1. access tuple elements
    - indexing
        ```python
        index(tup::Tuple[int], idx:int) -> int
        ```
        or
        ```python
        tup[0]
        ```
    - slicing
        ```python
        slice(tup::Tuple[int], start::int, step::int=0, end::int) -> Tuple[int]
        ```
        or
        ```python
        tup[1:2]
        ```
2. update tuple elements
    - Tuples are immutable so its value cannot be updated or changed.
    - A new tuple canbe created from a tuple.

    1. `add`
        ```python
        add(tup1::Tuple[int], tup2::Tuple[int]) ->Tuple[int]
        ```
        Example:
        ```python
        tup1 = (1, 2, 3)
        tup2 = (4, 5)
        tup1 + tup2 = (1, 2, 3, 4, 5)
        ```
    2. `insert_at`: insert an element into a tuple at a given position

        ```python
        insert(tup::Tuple, pos::int, value::int) -> Tuple[int]
        ```
        Example:
        ```python
        tup = (1, 2, 3)
        insert(tup, 1, 5)
        tup = (1, 5, 2, 3)
        ```
    3. `remove_at`: remove tuple elements by index
        ```python
        del(tup::Tuple, pos::int) -> Tuple[int]
        ```
        Example:
        ```python
        tup = (2, 3, 4)
        del(tup, 1)
        tup = (2, 4)
        ```


# Tensor

A tensor is characterized by (1) elemental type; (2) its shape which is the number of dimensions and items in a tensor.

## Tensor operations in neural networks

### Item access and manipulation

1. _**slice**_
    ```python
    slice(X:Tensor<real>, idx:int, dim:int) -> Tensor<real>
    ```
   - shape function: $S(\mathbf{Y}) = \tau(S(\mathbf{X}), \text{dim}, \text{keep\_dim})$
       - $\text{insert}(\text{del}(S(\mathbf{Y}), \text{dim}), \text{dim}, 1) \quad \text{if keep\_dim}$
       - $\text{del}(S(\mathbf{Y}), \text{dim}) \quad \text{otherwise}$

   - computation

### Neural network specializations

1. _**embedding**_: parallel slicing
    ```python
    embedding(X:Vector<int>, Y:Tensor<real>, dim:int) -> Z:Tensor<real>
    ```
    - shape function

        $$S(Z) = \tau (S(\mathbf{X}), S(\mathbf{Y}), \text{dim}) = (S(\mathbf{X})[0]) + \text{del}(S(\mathbf{Y}), \text{dim}) $$

    - computation

        $\text{foreach} \quad (i, x) \quad \text{in} \quad \mathbf{X}_{N}$

        $\quad \quad \text{slice}(Z, \text{dim}=i) = \text{slice}(\mathbf{Y}, \mathbf{X}[i], \text{dim})$
