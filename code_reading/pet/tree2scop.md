
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Tree to SCoP](#tree-to-scop)
    - [Entry point](#entry-point)
    - [Key data structures](#key-data-structures)
        - [pet_context](#petcontext)

<!-- /TOC -->

# Tree to SCoP

## Entry point

The entry point is in [scan](https://github.com/Meinersbur/pet/blob/master/scan.cc#L2594).

```c
struct pet_scop *PetScan::scan(Stmt *stmt){
  SourceManager &SM = PP.getSourceManager();
  unsigned start_off, end_off;
  pet_tree *tree;
  ...

  tree = extract(StmtRange(start, end), false, false, stmt);
  tree = add_kills(tree, kl.locals);
  return extract_scop(tree);
}
```

[extract_scop](https://github.com/Meinersbur/pet/blob/master/scan.cc#L2479)

```c
/* Add a call to __pencil_kill to the end of "tree" that kills
 * all the variables in "locals" and return the result.
 *
 * No location is added to the kill because the most natural
 * location would lie outside the scop.  Attaching such a location
 * to this tree would extend the scope of the final result
 * to include the location.
 */
 __isl_give pet_tree *PetScan::add_kills(__isl_take pet_tree *tree,
set<ValueDecl *> locals){
  int i;
  pet_expr *expr;
  pet_tree *kill, *block;
  set<ValueDecl *>::iterator it;

  if (locals.size() == 0)
      return tree;
  expr = pet_expr_new_call(ctx, "__pencil_kill", locals.size());
  i = 0;
  for (it = locals.begin(); it != locals.end(); ++it) {
    pet_expr *arg;
    arg = extract_access_expr(*it);
    expr = pet_expr_set_arg(expr, i++, arg);
  }
  kill = pet_tree_new_expr(expr);
  block = pet_tree_new_block(ctx, 0, 2);
  block = pet_tree_block_add_child(block, tree);
  block = pet_tree_block_add_child(block, kill);

  return block;
}
```

## Key data structures

### pet_context

```c
/* A pet_context represents the context in which a pet_expr
 * in converted to an affine expression.
 *
 * "domain" prescribes the domain of the affine expressions.
 *
 * "assignments" maps variable names to their currently known values.
 * Internally, the domains of the values may be equal to some prefix
 * of the space of "domain", but the domains are updated to be
 * equal to the space of "domain" before passing them to the user.
 *
 * If "allow_nested" is set, then the affine expression created
 * in this context may involve new parameters that encode a pet_expr.
 *
 * "extracted_affine" caches the results of pet_expr_extract_affine.
 * It may be NULL if no results have been cached so far and
 * it is cleared (in pet_context_cow) whenever the context is changed.
 */
struct pet_context {
  int ref;

  isl_set *domain;
  isl_id_to_pw_aff *assignments; // this is a hashmap.
  int allow_nested;

  pet_expr_to_isl_pw_aff *extracted_affine;
};
```
