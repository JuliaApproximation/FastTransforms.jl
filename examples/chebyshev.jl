#############
# This demonstrates the Chebyshev transform and inverse transform,
# explaining precisely the normalization and points
#############

# first kind points -> first kind polynomials
n = 20
p_1 = [sinpi((n-2k-1)/2n) for k=0:n-1]
f = exp.(p_1)
f̌ = chebyshevtransform(f; kind=1)

f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# second kind points -> first kind polynomials
p_2 = [cospi(k/(n-1)) for k=0:n-1]
f = exp.(p_2)
f̌ = chebyshevtransform(f; kind=2)

f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# first kind polynomials -> first kind points
ichebyshevtransform(f̌; kind=1) ≈ exp.(p_1)

# first kind polynomials -> second kind points
ichebyshevtransform(f̌; kind=2) ≈ exp.(p_2)
