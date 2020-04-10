<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Basics](#basics)
    - [Data structure of `pet_scop`](#data-structure-of-petscop)
        - [Data structure of `pet_array`](#data-structure-of-petarray)
        - [Data structure of `pet_stmt`](#data-structure-of-petstmt)

<!-- /TOC -->

# Basics

## Data structure of `pet_scop`

Call [pet_scop_extract_from_C_source](https://github.com/Meinersbur/pet/blob/a12f2f2f73/pet.cc#L1161) to parse scop from C source file.

```c
struct pet_scop *pet_scop_extract_from_C_source(isl_ctx *ctx,
    const char *filename, const char *function)
```

The data structure of pet scop.

```c
struct pet_scop {
  pet_loc *loc;

  // In the final result, the context describes the set of parameter values
  // for which the scop can be executed.
  // During the construction of the pet_scop, the context lives in a set space
  // where each dimension refers to an outer loop.
  isl_set *context;

  // describes assignments to the parameters (if any) outside of the scop.
  isl_set *context_value;

  isl_schedule *schedule;

  // define types that may be referenced from by the arrays.
  int n_type;
  struct pet_type **types;

  int n_array;
  struct pet_array **arrays;

  int n_stmt;
  struct pet_stmt **stmts;

  // describe implications on boolean filters
  int n_implication;
  struct pet_implication **implications;

  // describe independences implied by for loops that are marked independent
  // in the source code
  int n_independence;
  struct pet_independence **independences;
};
typedef struct pet_scop pet_scop;
```

isl uses six types of objects for representing sets and relations:

1. `isl_basic_set`
1. `isl_basic_map`
    - represent sets and relations that can be described as a conjunction of affine constraints.
1. `isl_set`
1. `isl_map`
    - represent unions of `isl_basic_sets` and `isl_basic_maps`, respectively.
    - however, all `isl_basic_sets` or `isl_basic_maps` in the union need to live in the _**same space**_.
1. `isl_union_set`
1. `isl_union_map`
    - represent unions of `isl_sets` or `isl_maps` in _**different spaces**_, where spaces are considered different if they have a different number of dimensions and/or different names.

The difference between sets and relations (maps, integer maps are binary relations between integer sets) is that sets have one set of variables, while relations have two sets of variables, input variables and output variables.

### Data structure of `pet_array`

```c
struct pet_array {
  // holds constraints on the parameter that ensure that
  // this array has a valid (i.e., non-negative) size
  isl_set *context;
  isl_set *extent;  // holds constraints on the indices
  isl_set *value_bounds;  // holds constraints on the elements of the array;

  char *element_type;
  int element_is_record;
  int element_size;
  int live_out;

  // if uniquely_defined is set then the array is written by a single access
  // such that any element that is ever read is known to be assigned
  // exactly once before the read
  int uniquely_defined;

  // declared is set if the array was declared somewhere inside the scop.
  int declared;

  // exposed is set if the declared array is visible outside the scop.
  int exposed;

  // outer is set if the type of the array elements is a record and
  // the fields of this record are represented by separate pet_array structures.
  int outer;
};
```

### Data structure of `pet_stmt`

```c
struct pet_stmt {
  pet_loc *loc;

  // If the statement has arguments, i.e., n_arg != 0, then "domain" is a wrapped map
  // mapping the iteration domain to the values of the arguments
  // for which this statement is executed.
  // Otherwise, it is simply the iteration domain.

  // If one of the arguments is an access expression that accesses
  // more than one element for a given iteration,
  // then the constraints on the value of this argument (encoded in "domain")
  // should be satisfied for all of those accessed elements.
  isl_set *domain;

  pet_tree *body; // A pet_tree represents an AST.

  unsigned n_arg;
  pet_expr **args;
};
```
