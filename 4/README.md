# Exercise 1

## Exercise 1(a)

There are $\frac{128}{32} = 4$ warps per block.

## Exercise 1(b)

There are $\frac{1024}{128} = 8$ blocks, so there are $8 \times 4 = 32$ warps.

## Exercise 1(c)

### Exercise 1(c)(i)

In each block, $3$ warps are active so $3 \times 8 = 24$ total warps are active.

### Exercise 1(c)(ii)

In each block, $2$ warps are divergent so $2 \times 8 = 16$ total warps are divergent.

### Exercise 1(c)(iii)

All threads are active so the SIMD efficiency is $\frac{32}{32} = 1$.

### Exercise 1(c)(iv)

Threads $32$ to $39$ are inactive, so $32 - 8 = 24$ threads active. The SIMD efficiency is $\frac{24}{32} = 0.75$.

### Exercise 1(c)(v)

Threads $104$ to $127$ are inactive, so $32 - 24 = 8$ threads are active. The SIMD efficiency is $\frac{8}{32} = 0.25$.

## Exercise 1(d)

### Exercise 1(d)(i)

Every warp in the grid is active so $24$ warps.

### Exercise 1(d)(ii)

Every warp in the grid is divergent so $24$ warps.

### Exercise 1(d)(iii)

Odd numbered threads are inactive, so $16$ threads are active. The SIMD efficiency $\frac{16}{32} = 0.5$.

## Exercise 1(e)

### Exercise 1(e)(i)

The loop runs either $3$, $4$ or $5$ iterations, so $3$ iterations have no divergence.

### Exercise 1(e)(ii)

There are $2$ iterations with divergence.

# Exercise 2

The grid will contain $\lceil \frac{2000}{512} \rceil = 4$ blocks, so there will be $4 \times 512 = 2048$ threads.

# Exercise 3

Only $1$ warp will have divergence on the boundary check.

# Exercise 4

The times spent waiting per thread are $1.0, 0.7, 0.0, 0.2, 0.6, 1.1, 0.4, 0.1$ for a total of $4.1$ seconds. The total execution time is $3 \times 8 = 24$, so the waiting fraction is $\frac{4.1}{24} \approx 0.17$.

# Exercise 5

This is not a good idea. Future GPUs might use warps that have more or less than 32 threads.

# Exercise 6

(c).

# Exercise 7

## Exercise 7(a)

There are $8 \times 128 = 1024$ threads, so the occupancy is $\frac{1024}{2048} = 0.5$.

## Exercise 7(b)

There are $16 \times 64 = 1024$ threads, so the occupancy is $\frac{1024}{2048} = 0.5$.

## Exercise 7(c)

There are $32 \times 32 = 1024$ threads, so the occupancy is $\frac{1024}{2048} = 0.5$.

## Exercise 7(d)

There are $64 \times 32 = 2048$ threads, so the occupancy is $\frac{2048}{2048} = 0.5$.

## Exercise 7(e)

There are $32 \times 64 = 2048$ threads, so the occupancy is $\frac{2048}{2048} = 0.5$.

# Exercise 8

## Exercise 8(a)

The kernel needs $\frac{2048}{128} = 16$ blocks and $2048 \times 30 = 61440$ registers, so it reaches maximum occupancy.

## Exercise 8(b)

The kernel needs $\frac{2048}{32} = 64$ blocks, so the kernel reaches half occupancy.

## Exercise 8(c)

The kernel needs $2048 \times 34 = 68632$ registers, so it reaches does not reach maximum occupancy. Registers for only $\frac{65536}{34} = 1927$ threads are available, so only $\frac{1927}{256} = 7$ blocks can be issued. This yields an occupancy of $\frac{7 \times 256}{2048} = 0.875$.

# Exercise 9

Each block would have $32 \times 32 = 1024$ threads which is not supported by the device.
