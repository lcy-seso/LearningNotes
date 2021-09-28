# [Firm-a graph-based intermediate representation](https://pp.ipd.kit.edu/uploads/publikationen/braun11wir.pdf)

##  1. <a name='Introduction'></a>Introduction

1. The heart of an intermediate representation is **operations**.
1. Analysis and transformation are the basic parts of any optimization pass.

Compiler optimizations can be classified into three kinds.

||Explicit|Implicit|Inherent (most desirable)|
|--|--|--|--|
|**What it is**| analyze and transform the program iteratively|not part of compiler phase|abstract away inessential aspects of the program during the IR construction.
|**Examples**|x*2 = x + x = x << 1|constant folding during construction of the IR|In a graph-based representation, dead code is not reachable by a walkover graph. By contrast, in a list-based representation, dead code will still be encountered. Dead-code elimination is inherent in a graph-based representation.|
|**Problems**|phase-ordering|local by nature thus can be performed in constant time|
|**Advantages**|powerful optimizations are in this kind|constant time|neither succumb to the phase-ordering nor limited to phrase ordering|

##  2. <a name='FIRM'></a>FIRM

FIRM is a concise graph-based representation:

1. represent program graphs as explicit dependency graphs in SSA-form.
    - even dependencies due to memory accesses are explicitly modeled.
1. Instructions are connected by dependency edges **relaxing the total to a partial order** inside a basic block.

# References
