<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [LIFT](#lift)
  - [Intermediate language](#intermediate-language)
    - [1. Algorithmic Patterns](#1-algorithmic-patterns)
    - [2. Data Layout Patterns](#2-data-layout-patterns)
    - [3. Parallel Patterns](#3-parallel-patterns)
    - [4. Address Space Patterns](#4-address-space-patterns)
    - [5. Vectorize Pattern](#5-vectorize-pattern)
- [Reference](#reference)

<!-- /TOC -->

# LIFT

- High-level languages based on parallel patterns capture rich information about the algorithmic structure of programs.
- The foundation of the Lift IL is lambda calculus which formalizes the reasoning about functions, their **composition**, **nesting** and application.

## Intermediate language

The Lift IL expresses program as compositions and nesting of functions which operate on <font color=#C71585> **arrays**</font>.

### 1. Algorithmic Patterns

1. mapSeq
1. reduceSeq
1. iterate

### 2. Data Layout Patterns

1. split
1. join
1. gather
1. scatter
1. zip
1. slide

### 3. Parallel Patterns

1. mapGlb
1. mapWrg
1. mapLcl

### 4. Address Space Patterns

1. toGlobal
1. toLocal
1. toPrivate

### 5. Vectorize Pattern

1. asVector
1. asScalar
1. mapVec

# Reference

1. Steuwer, Michel, Toomas Remmelg, and Christophe Dubach. "[Lift: a functional data-parallel IR for high-performance GPU code generation](https://eprints.gla.ac.uk/146596/1/146596.pdf)." 2017 IEEE/ACM International Symposium on Code Generation and Optimization (CGO). IEEE, 2017.
