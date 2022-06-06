<!-- vscode-markdown-toc -->

- [Towards Size-dependent types for array programming](#towards-size-dependent-types-for-array-programming)
  - [The F language](#the-f-language)
  - [To confirm](#to-confirm)
- [References](#references)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# [Towards Size-dependent types for array programming](https://futhark-lang.org/publications/array21.pdf)

## The F language

The simplicity of the language:

1. does not support type polymorphism.
    - functions are **type monomorphic** but **shape polymorphic**.
2. **the expressions are flatten** (do not quite understand this point)
    - expressions that allocate existentially sized arrays are bounded by explicit let-constructs.

## To confirm

- [ ] kinding
- [ ] let-polymorphism
- [ ] parametric size
- [ ] type-monomophism, type-polymorphism
- [ ] existential type
- [ ] bound variables, variable $x$ is bound in $\mu$
- [ ] type soundness
- [ ] well-formedness of types

# References