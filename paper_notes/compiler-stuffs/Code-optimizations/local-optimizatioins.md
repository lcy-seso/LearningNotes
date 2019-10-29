# Local Optimizations

Programmers will protest that they do not write code that contains redundant. In practice, redundancy elimination ﬁnds many opportunities. ***Translation from source code to ir elaborates many details, such as address calculations, and introduces redundant expressions***.

## Local Value Numbering (LVN)

***Local value numbering*** is one of the oldest and most powerful redundency elimination.
