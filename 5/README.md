# Exercise 1

No. Elements are used by only one thread.

# Exercise 2

N/a.

# Exercise 3 

For the first `__syncthreads()`, it is a write-after-read dependency. For the second `__syncthreads()`, it is a read-after-write dependency.

# Exercise 4

Using shared memory instead of registers can reduce the number of transfers between global memory and the execution unit. For example, if data needed by two threads is loaded directly into registers, two loads are required, but if the data is loaded into shared memory, only one load is required.

# Exercise 5

The reduction in bandwidth is $\frac{1}{32}$ because each element is used 32 times per load instead of only once per load.

# Exercise 6

One variable will be allocated per kernel. This yields $512 \times 1000 = 512000$ variables.

# Exercise 7

One variable will be allocated per block. This yields $1000$ variables.

# Exercise 8

## Exercise 8(a)

Let the input matrices be $A$ and $B$. Consider $A_{0,0}$. This element is used in the dot product between $A_{0,*}$ and $B_{*,i}$ for $0 \leq i < N$. Thus each element is loaded exactly $N$ times.

## Exercise 8(b)

Again, consider $A_{0, 0}$. This element will be loaded for the product of the top left tile of $A$ and each top row tile in $B$. There are $\frac{N}{T}$ such products. Thus each element is loaded exactly $\frac{N}{T}$ times.

# Exercise 9

## Exercise 9(a)

The kernel has a compute to global memory access ratio of $\frac{36}{7 \times 4} \approx 1.29$ FLOP/B. The device has a compute to global memory access ratio of $\frac{200}{100} = 2$ FLOP/B, so the kernel is memory-bound.

## Exercise 9(b)

The device has a compute to global memory access ratio of $\frac{300}{250} = 1.2$ FLOP/B, so the kernel is memory-bound.

# Exercise 10

## Exercise 10(a)

The kernel will only execute correctly when `BLOCK_SIZE == 1`. There is a read-after-write dependency between line 10 and 11.

## Exercise 10(b)

The programmer must insert `__syncthreads()` between line 10 and 11.

# Exercise 11

## Exercise 11(a)

There is one per thread or $1024$ versions.

## Exercise 11(b)

There is one per thread or $1024$ versions.

## Exercise 11(c)

There is one per block or $8$ versions.

## Exercise 11(d)

There is one per block or $8$ versions.

## Exercise 11(e)

The kernel uses $(128 + 1) \times 4 = 516$ bytes per block.

## Exercise 11(f)

There are 5 loads and 1 store to global memory. There are 5 multiply and 5 addition operations. This yields a compute to global memory access ratio of $\frac{10}{6} \approx 1.67$ FLOP/B.

# Exercise 12

## Exercise 12(a)

If the kernel uses the maximum number of threads per SM, then it uses $\frac{2048}{64} = 32$ blocks, $27 \times 2048 = 55296$ registers, and 4KB of shared memory per SM. Thus the kernel can reach full occupancy.

## Exercise 12(b)

If the kernel uses the maximum number of threads per SM, then it uses $\frac{2048}{256} = 8$ blocks, $31 \times 2048 = 63488$ registers, and 8KB of shared memory per SM. Thus the kernel can reach full occupancy.

