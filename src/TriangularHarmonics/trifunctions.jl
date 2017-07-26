function tri_zero_spurious_modes!(A::AbstractMatrix)
    M, N = size(A)
    @inbounds for j = 2:N
        @simd for i = M-j+2:M
            A[i,j] = 0
        end
    end
    A
end

function trirand{T}(::Type{T}, m::Int, n::Int)
    A = zeros(T, m, n)
    for j = 1:n
        for i = 1:m+1-j
            A[i,j] = rand(T)
        end
    end
    A
end

function trirandn{T}(::Type{T}, m::Int, n::Int)
    A = zeros(T, m, n)
    for j = 1:n
        for i = 1:m+1-j
            A[i,j] = randn(T)
        end
    end
    A
end

function triones{T}(::Type{T}, m::Int, n::Int)
    A = zeros(T, m, n)
    for j = 1:n
        for i = 1:m+1-j
            A[i,j] = one(T)
        end
    end
    A
end

trizeros{T}(::Type{T}, m::Int, n::Int) = zeros(T, m, n)

doc"""
Pointwise evaluation of triangular harmonic:

```math
\tilde{P}_{\ell,m}^{(\alpha,\beta,\gamma)}(x,y).
```
"""
trievaluate(x, y, L, M, α, β, γ) = trievaluate(x, L, M, α, β, γ)*trievaluate(x, y, M, β, γ)

function trievaluate(x::Number, L::Integer, M::Integer, α::Number, β::Number, γ::Number)

end

function trievaluate(x::Number, y::Number, M::Integer, β::Number, γ::Number)

end
