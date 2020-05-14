<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [scan_scops](#scanscops)
- [scan](#scan)
- [extract](#extract)

<!-- /TOC -->

1. Pet implement a Clang [AST consumer](https://github.com/Meinersbur/pet/blob/master/pet.cc#L568);
1. Register callback to handle [DeclGroupRef](https://github.com/Meinersbur/pet/blob/master/pet.cc#L695).
    - clang doc for [DeclGroupRef](https://clang.llvm.org/doxygen/classclang_1_1DeclGroupRef.html).
1. In the callback, [scan_scops](https://github.com/Meinersbur/pet/blob/master/pet.cc#L723)
 is responsible for extracting pet_scop.

# scan_scops

https://github.com/Meinersbur/pet/blob/master/scan.cc#L3308

```c
pet_scop *scop;
Stmt *stmt;

stmt = fd->getBody();

current_line = loc.start_line;
scop = scan(stmt);
scop = pet_scop_update_start_end(scop, loc.start, loc.end);

scop = add_parameter_bounds(scop);
scop = pet_scop_gist(scop, value_bounds);
```

# scan
https://github.com/Meinersbur/pet/blob/master/scan.cc#L2545

1. collect locals that will be killed after leaving the SCoP (???).
1. extract [pet_tree](https://github.com/Meinersbur/pet/blob/master/tree.h#L48) while traversing Clang AST: [extract](https://github.com/Meinersbur/pet/blob/master/scan.cc#L2193)
1. [add kills](https://github.com/Meinersbur/pet/blob/master/scan.cc#L2593)
1. extract scop from pet_tree: [struct pet_scop *PetScan::extract_scop(__isl_take pet_tree *tree)](https://github.com/Meinersbur/pet/blob/master/scan.cc#L2479)

# extract

https://github.com/Meinersbur/pet/blob/master/scan.cc#L2193

```c
__isl_give pet_tree *PetScan::extract(
  StmtRange stmt_range, bool block,
    bool skip_declarations, Stmt *parent){

    StmtIterator i;
    int j;

    for (i = stmt_range.first, j = 0; i != stmt_range.second; ++i, ++j)
        ;

  // j = number of statement in the given range; number of children;
    tree = pet_tree_new_block(ctx, block, j);

    skip = 0;
    for (; i != stmt_range.second; ++i) {
        // iterate over each statement in the given range;
        Stmt *child = *i;
        pet_tree *tree_i;

        // recursion;
        tree_i = extract(child);
    tree = pet_tree_block_add_child(tree, tree_i);
  }

    return tree;
}
```
