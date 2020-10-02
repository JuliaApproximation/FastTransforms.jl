# # Chebyshev transform
# This demonstrates the Chebyshev transform and inverse transform,
# explaining precisely the normalization and points

using FastTransforms
n = 20

# First kind points $\to$ first kind polynomials
p_1 = chebyshevpoints(Float64, n, Val(1))
f = exp.(p_1)
f̌ = chebyshevtransform(f, Val(1))
f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# First kind polynomials $\to$ first kind points
ichebyshevtransform(f̌, Val(1)) ≈ exp.(p_1)

# Second kind points $\to$ first kind polynomials
p_2 = chebyshevpoints(Float64, n, Val(2))
f = exp.(p_2)
f̌ = chebyshevtransform(f, Val(2))
f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# First kind polynomials $\to$ second kind points
ichebyshevtransform(f̌, Val(2)) ≈ exp.(p_2)

# First kind points $\to$ second kind polynomials
p_1 = chebyshevpoints(Float64, n, Val(1))
f = exp.(p_1)
f̌ = chebyshevutransform(f, Val(1))
f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-1]' * f̌
f̃(0.1) ≈ exp(0.1)

# Second kind polynomials $\to$ first kind points
ichebyshevutransform(f̌, Val(1)) ≈ exp.(p_1)

# Second kind points $\to$ second kind polynomials
p_2 = chebyshevpoints(Float64, n, Val(2))[2:n-1]
f = exp.(p_2)
f̌ = chebyshevutransform(f, Val(2))
f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-3]' * f̌
f̃(0.1) ≈ exp(0.1)

# Second kind polynomials $\to$ second kind points
ichebyshevutransform(f̌, Val(2)) ≈ exp.(p_2)
