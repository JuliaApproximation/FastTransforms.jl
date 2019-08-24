#############
# This demonstrates the Chebyshev transform and inverse transform,
# explaining precisely the normalization and points
#############

using FastTransforms

# first kind points -> first kind polynomials
n = 20
p_1 = chebyshevpoints(Float64, n; kind=1)
f = exp.(p_1)
f̌ = chebyshevtransform(f; kind=1)

f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# first kind polynomials -> first kind points
ichebyshevtransform(f̌; kind=1) ≈ exp.(p_1)

# second kind points -> first kind polynomials
p_2 = chebyshevpoints(Float64, n; kind=2)
f = exp.(p_2)
f̌ = chebyshevtransform(f; kind=2)

f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# first kind polynomials -> second kind points
ichebyshevtransform(f̌; kind=2) ≈ exp.(p_2)


# first kind points -> second kind polynomials
n = 20
p_1 = chebyshevpoints(Float64, n; kind=1)
f = exp.(p_1)
f̌ = chebyshevutransform(f; kind=1)
f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# second kind polynomials -> first kind points
ichebyshevutransform(f̌; kind=1) ≈ exp.(p_1)


# second kind points -> second kind polynomials
p_2 = chebyshevpoints(Float64, n; kind=2)[2:n-1]
f = exp.(p_2)
f̌ = chebyshevutransform(f; kind=2)
f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-3]' * f̌
f̃(0.1) ≈ exp(0.1)

# second kind polynomials -> second kind points
ichebyshevutransform(f̌; kind=2) ≈ exp.(p_2)
