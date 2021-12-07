# Arrow head types used in MixedGraph and classes that inherent from it, e.g., partially directed graphs, CPDAG and PAG
Undirected = '---'  # X---Y (for CPDAG and PAG)
Directed = '<--'  # X-->Y (for CPDAG and PAG)
Circle = 'o--'  # X--*Y  (for PAGs)
Tail = Undirected

# In PAGs there are 6 edge types: o--o, o---, o-->, --->, <-->, ----. In MAGs: --->, <-->, ----
