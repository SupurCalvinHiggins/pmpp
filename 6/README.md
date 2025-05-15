# Exercise 1

See `1.cu`.

# Exercise 2

Assuming no alignment requirements and that padded threads do not count against coalescing, any block size will work.

# Exercise 3

## Exercise 3(a)

Coalesced.

## Exercise 3(b)

Not applicable.

## Exercise 3(c)

Coalesced.

## Exercise 3(d)

Uncoalesced.

## Exercise 3(e)

Not applicable.

## Exercise 3(f)

Not applicable.

## Exercise 3(g)

Coalesced.

## Exercise 3(h)

Coalesced.

## Exercise 3(i)

Uncoalesced.

# Exercise 4

## Exercise 4(a)

Each thread performs $2n$ loads, $1$ store and $2n$ FLOPs. The FLOP/B ratio is $\frac{8n}{2n + 1} \approx 4$.

## Exercise 4(b)

Each thread performs loads from $2 \times \frac{n}{32}$ tiles, $1$ store, and $2n$ FLOPs. The FLOP/B ratio is $\frac{8n}{\frac{n}{16} + 1} \approx 128$.

## Exercise 4(c)

Thread coarsening does not change the FLOP/B ratio so it is still $\approx 128$.

