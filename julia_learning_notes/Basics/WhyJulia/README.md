1. [Why Julia](https://ucidatascienceinitiative.github.io/IntroToJulia/Html/WhyJulia)
1. [Notes](whyJulia.pdf)

---

### What is [Type-stable](https://docs.julialang.org/en/v1.2-dev/manual/faq/#man-type-stability-1)

_**The type of the output is predictable from the types of the inputs.**_ In particular, it means that the type of the output cannot vary depending on the values of the inputs.

The following codes are type-unstable:

```julia
function unstable(flag::Bool)
  if flag
    return 1
  else
    return 1.0
  end
end
```

_**Julia can't predict the return type of function that is type unstable at compile-time, making generation of fast machine code difficult.**_
