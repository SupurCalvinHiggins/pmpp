# Exercise 1

## Exercise 1(a)

See `1a.cu`.

## Exercise 1(b)

See `1b.cu`.

## Exercise 1(c)

Performance will depend on the shape of the output matrix. If it has many rows and few columns, the first kernel will be better. If it has few rows and many columns, the second kernel will be better.

# Exercise 2

See `2.cu`.

# Exercise 3

## Exercise 3(a)

There are $16 \times 32 = 512$ threads per block.

## Exercise 3(b)

There are $\frac{300 - 1}{16} + 1 = 19$ rows and $\frac{150 - 1}{32} + 1 = 5$ columns
of blocks. Since there are $512$ threads per block, there are 
$19 \times 5 \times 512 = 48640$ threads in the grid.

## Exercise 3(c)

There are $19 \times 5 = 95$ blocks in the grid.

## Exercise 3(d)

There are $300 \times 150 = 45000$ threads that execute line 5.

# Exercise 4

## Exercise 4(a)

The array index is $20 \times 400 + 10 = 8010$.

## Exercise 4(b)

The array index is $500 \times 10 + 20 = 5020$.

# Exercise 5

The array index is $10 + 400 \times (20 + 500 \times 5) = 1008010$.
