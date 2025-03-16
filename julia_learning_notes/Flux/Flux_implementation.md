<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Notes for Flux.jl](#notes-for-fluxjl)
    - [Overall information](#overall-information)
    - [How Flux works](#how-flux-works)
        - [Macro `@grad`](#macro-grad)
        - [Function `track`](#function-track)
    - [Codes Snippets](#codes-snippets)
        - [1. Construct the Tape](#1-construct-the-tape)
        - [2. Track the Computations](#2-track-the-computations)
        - [3. Play the Tape in Forward Order and Replay It in Reverse Order](#3-play-the-tape-in-forward-order-and-replay-it-in-reverse-order)

<!-- /TOC -->

# Notes for Flux.jl

## Overall information

[Flux.jl](https://github.com/FluxML/Flux.jl) implements the Tape based reverse-mode auto differentiation. [This note]( https://github.com/lcy-seso/learning_notes/blob/master/paper_notes/AD/brief_introduction_to_AD.md)  gives some basic concepts about AD. The implementation challenges for reverse-mode mode is: the results of forwarding computations(including the operation itself) _**have to be recorded and maintained**_ until the corresponding gradient computation is finished.

1. Flux's implementation can be characterized as <span style="background-color:#ACD6FF;">_**a traditional tracing based approach**_</span>.
1. Nevertheless, Flux's implementation is <span style="background-color:#ACD6FF;">a good example to study the Julia programming language's _**meta programming**_, _**types**_, and _**multiple dispatch**_.</span>
1. Currently, Flux only supports gradient calculation for scalar, vector, matrix computation. It does not support computations for Tensors whose rank are more than two. But this canbe extended.

The implementations are mainly made up of two parts:

1. Types and utilities for constructing "Tape" while the user defines his forward computation.
    * `Tracked*` series type definitions
    * `@grad` macro
    * `track` function
1. Play the tape in forwarding order and replay it in the reverse order to calculate the gradients.
    * `back` series function

## How Flux works

The very high-level idea of AD in Flux is:

1. If the user needs to calculate gradients of an n-d array. It has to be defined as a `TrackedReal`, `TrackedArray`, or some `Tracked*` type.
    _**The `Tracked*` type wraps Julia's built-in `Array` type. It can be regarded as an item in Tape.**_
    * If a value whose type if Julia's built-in `Array` computes with a vale whose type is `Tracked*`, the result has a `Tracked*` type.
    * An value with a `Tracked*`typed compute with a value with a `Tracked*` type will lead to a result with a `Tracked*` type.
1. New methods that take `Tracked*` typed values as inputs have to be added for all differentiable operations which are implemented by using [Julia's code generation](https://docs.julialang.org/en/v1/manual/metaprogramming/index.html#Code-Generation-1) capabilities.
1. Every `Tracked*` object is a record on the Tape. Finally, call `back` function to replay the Tape to perform gradient computations.

### Macro `@grad`

`@grad` macro takes a function and its gradient function as its inputs, and _**generate**_ the internal function: `Tracker._forward`.

1. `Tracker._forward` is not exported to be used by users.
1. `Tracker._forward` is generated using macro `@grad`.
1. `Tracker._forward` can be regarded as a "lookup" operation that receives the forward computation function as its input (and also the inputs) and returns the corresponding backpropagator.
1. _**A function `f` in Julia has a singleton type `f`**_ (name of this type is the same as function's name). The `@grad` macro "generate" methods of `Tracker._forward` for every differentiable operation supported by Flux.
1. `Tracker._forward` returns:
    1. the forward computation results whose type is Julia's built-in Array type.
    1. the back propagator (the function that calculates gradients).

### Function `track`

Function `track` is an exported function in Flux which warps the original forward function and construct a `Tracked*` computation results.

_**`track` calculate the forward computation result and "looks ups" the corresponding backpropagator by invoking `Tracker._forward` internally.**_

## Codes Snippets

Tracker in Flux is a [Julia module](https://docs.julialang.org/en/v1/manual/modules/#modules-1) which is a separate variable workspace.

### 1. Construct the Tape

In the `Tracked*` series type, three important information is recorded in the `tracker` field:

1. result of the forward computation
1. input parameters of forward computation
1. backpropagator that calculates the gradients

Important types to implement gradient calculation:

1. <span style="background-color:#DDA0DD;">_**[Internal Type]**_</span> The `Call` type: _**this is the type for generated backpropagator**_

    ```julia {.line-numbers}
    struct Call{F,As<:Tuple}
        func::F
        args::As
    end

    # Cor
    Call(f::F, args::T) where {F,T} = Call{F,T}(f, args)
    Call() = Call(nothing, ())
    ```

1. <span style="background-color:#DDA0DD;">_**[Exported Type]**_</span> `Tracker*` (`TrackedReal`, `TrakcedVector`, `TrackedMatrix`, ...)
    * [TrackedReal](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/scalar.jl#L1)
      ```julia {.line-numbers}
      struct TrackedReal{T<:Real} <: Real
        data::T  # this is the forward computation result
        tracker::Tracked{T}
      end
      ```
    * [TrackedArray](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/array.jl#L9)

        ```julia {.line-numbers}
        struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
          tracker::Tracked{A}
          data::A
          grad::A

          # Cor
          TrackedArray{T,N,A}(t::Tracked{A}, data::A) where {T,N,A} = new(t, data)
          TrackedArray{T,N,A}(t::Tracked{A}, data::A, grad::A) where {T,N,A} = new(t, data, grad)
        end
        ```
1. <span style="background-color:#DDA0DD;">_**[Internal Type]**_</span> [Tracked](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/Tracker.jl#L31) which is a [mutable struct](https://docs.julialang.org/en/v1/manual/types/#Mutable-Composite-Types-1w).

    _**A `Tracked` object holds forward computation results, the backpropagator, and memory for gradients.**_

    ```julia {.line-numbers}
    mutable struct Tracked{T}
      ref::UInt32
      f::Call    # this field is the backpropagator
      isleaf::Bool
      grad::T    # This is used to stored gradients
                 # but why there is another `grad` in TrackedArray?
                 # It seems that `grad` in TrackedArray is not used?

      # Cor
      Tracked{T}(f::Call) where T = new(0, f, false)
      Tracked{T}(f::Call, grad::T) where T = new(0, f, false, grad)
      Tracked{T}(f::Call{Nothing}, grad::T) where T = new(0, f, true, grad)
    end
    ```

    _**Construct a `Tracked` object**_

    1. [track(f::F, xs...; kw...)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/Tracker.jl#L49): ~~this function is used to regists a customized operator and its differential function.~~ this function construct a `Tracked` object.

        ```julia {.line-numbers}
        function track(f::F, xs...; kw...) where F
          # y is the output of the forward computation
          # back is the gradient function that is not implemented by users
          # but is generated by Flux's codes.
          y, back = _forward(f, xs...; kw...)
          track(Call(back, tracker.(xs)), y)
        end
        ```

        ```julia {.line-numbers}
        tracker(x::TrackedArray) = x.tracker  # x.tracker is a TrackedArray.
        ```

        line 6 call the below `track` to construct a `Tracked` object.
        ```julia {.line-numbers}
        track(f::Call, x) = Tracked{typeof(x)}(f)
        ```

        _**PLEASE NOTE the `track` function can also construct a `Tracked*` type.**_

        ```julia
        track(c::Call, x::AbstractArray) = TrackedArray(c, x)
        ```

    2. Construct a new `Tracked*` object from a `Tracked*` object using `param`:

        ```julia {.line-numbers}
        @grad identity(x) = data(x), Δ -> (Δ,)
        param(x::TrackedReal) = track(identity, x)
        param(x::TrackedArray) = track(identity, x)
        ```
### 2. Track the Computations

_**Define Tracked\* objects**_: <span style="background-color:#DDA0DD;">Every array expected to differentiate has to be defined as a `Tracked*` type</span> using [param](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/Tracker.jl#L105) interface.

1. `Tracked*` constructed from Julia's built-in `Array` type.

    ```julia {.line-numbers}
    param(x::Number) = TrackedReal(float(x))
    param(xs::AbstractArray) = TrackedArray(float.(xs))

    # the input variable has no gradient functions and
    # the gradients are initialized to zeros.
    TrackedArray(x::AbstractArray) = TrackedArray(Call(), x, zero(x))
    TrackedArray(c::Call, x::A, Δ::A) where A <: AbstractArray =
        TrackedArray{eltype(A),ndims(A),A}(Tracked{A}(c, Δ), x, Δ)
    ```
1. For every supported differentiable function `f` in Flux, a new method whose input/inputs is/are `Tracked*` is added.

    Let's take `Base.maximum` for example:

    ```julia {.line-numbers}
    import Base: *

    track(c::Call, x::AbstractArray) = TrackedArray(c, x)
    # add method for Base.maximum
    Base.maximum(xs::TrackedArray; dims = :) = track(maximum, xs, dims = dims)
    ```

### 3. Play the Tape in Forward Order and Replay It in Reverse Order

Backpropagation begins from the training loop in [train.jl](https://github.com/FluxML/Flux.jl/blob/master/src/optimise/train.jl#L55):

```julia {.line-numbers}
function train!(loss, data, opt; cb = () -> ())
  ...
  @progress for d in data
    l = loss(d...)
    @interrupts back!(l)
    opt()
    cb()
end
```

Line 5, function `back!` calculates the gradients. Let's look into implementation details of `back!`

1. `scalar.jl` --> [back!(x::TrackedReal)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/lib/real.jl#L13)
    ```julia {.line-numbers}
    function back!(x::TrackedReal)
      isinf(x) && error("Loss is Inf")
      isnan(x) && error("Loss is NaN")
      return back!(x, 1)
    end
    ```
    * Line 2 ~ 3 is a [short-circuit evaluation](https://docs.julialang.org/en/v1/manual/control-flow/#Short-Circuit-Evaluation-1).

1. `back.jl` --> [back!(x, Δ))](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L62)
    ```julia {.line-numbers}
    function back!(x, Δ)
      istracked(x) || return
      scan(x)  # initialize gradients to zeros
      back(tracker(x), Δ)
      return
    end
    ```
    * Let's look into the implmentation of [scan](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L7):

        ```julia {.line-numbers}
        function scan(x)  # here type of x is a TrackedReal
            istracked(x) && scan(tracker(x))
            return
        end
        ```

      ```julia {.line-numbers}
      function scan(x::Tracked)
        x.isleaf && return  # return if x is a leaf.
        ref = x.ref += 1
        if ref == 1
          scan(x.f)
          # initialize gradients to zeors if x is a leaf
           # and reference count is equal to 1
          isdefined(x, :grad) && (x.grad = zero_grad!(x.grad))
        end
        return
      end
      ```
      line 5 is another method of `scan`:
      ```julia {.line-numbers}
      scan(c::Call) = foreach(scan, c.args)
      ```
1. `back.jl` --> [back(x::Tracked, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L35)
    ```julia {.line-numbers}
    accum!(x, Δ) = x .+ Δ
    accum!(x::AbstractArray, Δ) = (x .+= Δ)
    ```

    ```julia {.line-numbers}
    function back(x::Tracked, Δ)
      x.isleaf && (x.grad = accum!(x.grad, Δ); return)
      ref = x.ref -= 1
      if ref > 0 || isdefined(x, :grad)
        if isdefined(x, :grad)
          x.grad = accum!(x.grad, Δ)
        else
          x.grad = Δ
        end
        ref == 0 && back_(x.f, x.grad)
      else
        ref == 0 && back_(x.f, Δ)
      end
      return
    end
    ```

      line 10 calls the below `back`:

    * [back_(c::Call, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L22)
        ```julia {.line-numbers}
        function back_(c::Call, Δ)
          Δs = c.func(Δ)
          (Δs isa Tuple && length(Δs) >= length(c.args)) ||
            error("Gradient is not a tuple of length $(length(c.args))")
          foreach(back, c.args, data.(Δs))
        end
        ```

---

_**[TODO]**_ Not sure when below `back` functions are called.

* [back(g::Grads, x::Tracked, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L111)

    ```julia {.line-numbers}
    accum!(g::Grads, x, Δ) = g[x] = haskey(g, x) ? g[x] .+ Δ : Δ
    ```

    ```julia {.line-numbers}
    function back(g::Grads, x::Tracked, Δ)
      x.isleaf && (accum!(g, x, Δ); return)
      ref = x.ref -= 1
      if ref > 0 || haskey(g, x)
        accum!(g, x, Δ)
        ref == 0 && back_(g, x.f, g[x])
      else
        ref == 0 && back_(g, x.f, Δ)
      end
      return
    end
    ```

* [back_(g::Grads, c::Call, Δ)](https://github.com/FluxML/Flux.jl/blob/master/src/tracker/back.jl#L102)

    ```julia {.line-numbers}
    function back_(g::Grads, c::Call, Δ)
      Δs = c.func(Δ)
      (Δs isa Tuple && length(Δs) >= length(c.args)) ||
        error("Gradient is not a tuple of length $(length(c.args))")
      foreach((x, Δ) -> back(g, x, Δ), c.args, Δs)
    end
    ```
