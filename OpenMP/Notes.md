<h1>Diagonal</h1>
<p>Data-Sharing Attributes in This Region

1. left, right, result
Status: shared
Why:
They are function arguments.
They are defined outside the parallel region.
OpenMP’s default rule: variables visible before a parallel region are shared.
Implication:
All threads see the same matrices.
This is safe here because:
left and right are read-only.
result(i,j) is written by exactly one thread (unique i per thread).
No race condition exists</p>

2. N
Status: shared
Why:
Defined before the parallel for.
Not modified inside the region.
Implication:
All threads read the same loop bound.
This is safe.

3. Loop variable i
Status: private (implicitly)
Why:
OpenMP automatically privatizes the loop control variable of a parallel for.
Implication:
Each thread has its own copy of i.
Iterations of the outer loop are divided among threads.
No two threads execute the same value of i.

4. Loop variable j
Status: private (implicitly)
Why:
j is declared inside the parallel region (inside the loop body).
Variables declared inside a parallel region are private by default.
Implication:
Each thread has its own j.
No race condition.


