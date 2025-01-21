abstract type AbstractGramMatrix{T} <: LayoutMatrix{T} end

@inline issymmetric(G::AbstractGramMatrix) = true
@inline isposdef(G::AbstractGramMatrix) = true
@inline colsupport(G::AbstractGramMatrix, j) = colrange(G, j)

"""
    GramMatrix(W::AbstractMatrix, X::AbstractMatrix)

Construct a symmetric positive-definite Gram matrix with data stored in ``W``.
Given a family of orthogonal polynomials ``𝐏(x) = {p₀(x), p₁(x),…}``
and a continuous inner product ``⟨f, g⟩``, the Gram matrix is defined by:
```math
Wᵢⱼ = ⟨pᵢ₋₁, pⱼ₋₁⟩.
```
Moreover, given ``X``, the transposed Jacobi matrix that satisfies ``x 𝐏(x) = 𝐏(x) X``,
the Gram matrix satisfies the skew-symmetric rank-2 displacement equation (``X = X[1:n, 1:n]``):
```math
XᵀW - WX = GJGᵀ,
```
where ``J = [0 1; -1 0]`` and where:
```math
G[:, 1] = 𝐞_n, G_{:, 2} = W[n-1, :]X[n-1, n] - Xᵀ W[:, n].
```
Fast (``O(n^2)``) Cholesky factorization of the Gram matrix returns the
connection coefficients between ``𝐏(x)`` and the polynomials ``𝐐(x)``
orthogonal in the modified inner product, ``𝐏(x) = 𝐐(x) R``.
"""
struct GramMatrix{T, WT <: AbstractMatrix{T}, XT <: AbstractMatrix{T}} <: AbstractGramMatrix{T}
    W::WT
    X::XT
    function GramMatrix{T, WT, XT}(W::WT, X::XT) where {T, WT, XT}
        if size(W) ≠ size(X)
            throw(ArgumentError("Cannot construct a GramMatrix with W and X of different sizes."))
        end
        if !issymmetric(W)
            throw(ArgumentError("Cannot construct a GramMatrix with a nonsymmetric W."))
        end
        if bandwidths(X) ≠ (1, 1)
            throw(ArgumentError("Cannot construct a GramMatrix with a nontridiagonal X."))
        end
        new{T, WT, XT}(W, X)
    end
end

GramMatrix(W::WT, X::XT) where {T, WT <: AbstractMatrix{T}, XT <: AbstractMatrix{T}} = GramMatrix{T, WT, XT}(W, X)

@inline size(G::GramMatrix) = size(G.W)
@inline getindex(G::GramMatrix, i::Integer, j::Integer) = G.W[i, j]
@inline bandwidths(G::GramMatrix) = bandwidths(G.W)
@inline MemoryLayout(G::GramMatrix) = MemoryLayout(G.W)

"""
    GramMatrix(μ::AbstractVector, X::AbstractMatrix)

Construct a GramMatrix from modified orthogonal polynomial moments and the multiplication operator.
In the standard (classical) normalization, ``p_0(x) = 1``, so that the moments
``\\mu_n = ⟨ p_{n-1}, 1⟩`` are in fact the first column of the Gram matrix.
The recurrence is built from ``X^\\top W = WX``.
"""
GramMatrix(μ::AbstractVector{T}, X::XT) where {T, XT <: AbstractMatrix{T}} = GramMatrix(μ, X, one(T))
function GramMatrix(μ::AbstractVector{T}, X::XT, p0::T) where {T, XT <: AbstractMatrix{T}}
    N = length(μ)
    n = (N+1)÷2
    @assert N == size(X, 1) == size(X, 2)
    @assert bandwidths(X) == (1, 1)
    W = LowerTriangular(Matrix{T}(undef, N, N))
    if n > 0
        @inbounds for m in 1:N
            W[m, 1] = p0*μ[m]
        end
    end
    if n > 1
        @inbounds for m in 2:N-1
            W[m, 2] = (X[m-1, m]*W[m-1, 1] + (X[m, m]-X[1, 1])*W[m, 1] + X[m+1, m]*W[m+1, 1])/X[2, 1]
        end
    end
    @inbounds @simd for n in 3:n
        for m in n:N-n+1
            W[m, n] = (X[m-1, m]*W[m-1, n-1] + (X[m, m]-X[n-1, n-1])*W[m, n-1] + X[m+1, m]*W[m+1, n-1] - X[n-2, n-1]*W[m, n-2])/X[n, n-1]
        end
    end
    return GramMatrix(Symmetric(W[1:n, 1:n], :L), eval(XT.name.name)(view(X, 1:n, 1:n)))
end

function GramMatrix(μ::PaddedVector{T}, X::XT, p0::T) where {T, XT <: AbstractMatrix{T}}
    N = length(μ)
    b = length(μ.args[2])-1
    n = (N+1)÷2
    @assert N == size(X, 1) == size(X, 2)
    @assert bandwidths(X) == (1, 1)
    W = BandedMatrix{T}(undef, (N, N), (b, 0))
    if n > 0
        @inbounds for m in 1:min(N, b+1)
            W[m, 1] = p0*μ[m]
        end
    end
    if n > 1
        @inbounds for m in 2:min(N-1, b+2)
            W[m, 2] = (X[m-1, m]*W[m-1, 1] + (X[m, m]-X[1, 1])*W[m, 1] + X[m+1, m]*W[m+1, 1])/X[2, 1]
        end
    end
    @inbounds @simd for n in 3:n
        for m in n:min(N-n+1, b+n)
            W[m, n] = (X[m-1, m]*W[m-1, n-1] + (X[m, m]-X[n-1, n-1])*W[m, n-1] + X[m+1, m]*W[m+1, n-1] - X[n-2, n-1]*W[m, n-2])/X[n, n-1]
        end
    end
    return GramMatrix(Symmetric(W[1:n, 1:n], :L), eval(XT.name.name)(view(X, 1:n, 1:n)))
end

#
# X'W-W*X = G*J*G'
# This returns G, where J = [0 1; -1 0], respecting the skew-symmetry of the right-hand side.
#
function compute_skew_generators(W::GramMatrix{T}) where T
    X = W.X
    n = size(W, 1)
    G = zeros(T, n, 2)
    G[n, 1] = one(T)
    G[:, 2] .= W[:, n-1]*X[n-1, n] + W[:, n]*X[n, n] - X'W[:, n]
    return G
end

function cholesky(W::GramMatrix{T}) where T
    cholesky(MemoryLayout(W), W)
end

function cholesky(_, W::GramMatrix{T}) where T
    n = size(W, 1)
    G = compute_skew_generators(W)
    L = zeros(T, n, n)
    c = W[:, 1]
    ĉ = zeros(T, n)
    l = zeros(T, n)
    v = zeros(T, n)
    row1 = zeros(T, n)
    fastcholesky!(L, W.X, G, c, ĉ, l, v, row1, n)
    return Cholesky(L, 'L', 0)
end

function fastcholesky!(L::Matrix{T}, X, G, c, ĉ, l, v, row1, n) where T
    @inbounds @simd for k in 1:n-1
        d = sqrt(c[k])
        for j in k:n
            L[j, k] = l[j] = c[j]/d
        end
        for j in k:n
            v[j] = G[j, 1]*G[k, 2] - G[j, 2]*G[k, 1]
        end
        for j in k+1:n-1
            ĉ[j] = (X[j-1, j]*c[j-1] + (X[j, j]-X[k, k])*c[j] + X[j+1, j]*c[j+1] + c[k]*row1[j] - row1[k]*c[j] - v[j])/X[k+1, k]
        end
        ĉ[n] = (X[n-1, n]*c[n-1] + (X[n, n]-X[k, k])*c[n] + c[k]*row1[n] - row1[k]*c[n] - v[n])/X[k+1, k]
        cst = X[k+1, k]/d
        for j in k+1:n
            row1[j] = -cst*l[j]
        end
        cst = c[k+1]/d
        for j in k:n
            c[j] = ĉ[j] - cst*l[j]
        end
        gd1 = G[k, 1]/d
        gd2 = G[k, 2]/d
        for j in k:n
            G[j, 1] -= l[j]*gd1
            G[j, 2] -= l[j]*gd2
        end
    end
    L[n, n] = sqrt(c[n])
end

function cholesky(::Union{AbstractBandedLayout, SymmetricLayout{<: AbstractBandedLayout}}, W::GramMatrix{T}) where T
    n = size(W, 1)
    G = compute_skew_generators(W)
    L = BandedMatrix{T}(undef, (n, n), (bandwidth(W, 1), 0))
    c = W[:, 1]
    ĉ = zeros(T, n)
    l = zeros(T, n)
    v = zeros(T, n)
    row1 = zeros(T, n)
    fastcholesky!(L, W.X, G, c, ĉ, l, v, row1, n)
    return Cholesky(L, 'L', 0)
end

function fastcholesky!(L::BandedMatrix{T}, X, G, c, ĉ, l, v, row1, n) where T
    b = bandwidth(L, 1)
    @inbounds @simd for k in 1:n-1
        d = sqrt(c[k])
        for j in k:min(k+b, n)
            L[j, k] = l[j] = c[j]/d
        end
        for j in max(k, n-b-1):n
            v[j] = G[j, 1]*G[k, 2] - G[j, 2]*G[k, 1]
        end
        for j in k+1:min(k+b+1, n-1)
            ĉ[j] = (X[j-1, j]*c[j-1] + (X[j, j]-X[k, k])*c[j] + X[j+1, j]*c[j+1] + c[k]*row1[j] - row1[k]*c[j] - v[j])/X[k+1, k]
        end
        if k ≥ n-b-1
            ĉ[n] = (X[n-1, n]*c[n-1] + (X[n, n]-X[k, k])*c[n] + c[k]*row1[n] - row1[k]*c[n] - v[n])/X[k+1, k]
        end
        cst = X[k+1, k]/d
        for j in k+1:min(k+b+1, n)
            row1[j] = -cst*l[j]
        end
        cst = c[k+1]/d
        for j in k:min(k+b+1, n)
            c[j] = ĉ[j] - cst*l[j]
        end
        gd1 = G[k, 1]/d
        gd2 = G[k, 2]/d
        for j in max(k, n-b-1):n
            G[j, 1] -= l[j]*gd1
            G[j, 2] -= l[j]*gd2
        end
    end
    L[n, n] = sqrt(c[n])
end

"""
    ChebyshevGramMatrix(μ::AbstractVector)

Construct a Chebyshev--Gram matrix of size `(length(μ)+1)÷2` with entries:
```math
W_{i,j} = \\frac{\\mu_{|i-j|+1} +\\mu_{i+j-1}}{2}.
```
Due to the linearization of a product of two first-kind Chebyshev polynomials,
the Chebyshev--Gram matrix can be constructed from modified Chebyshev moments:
```math
\\mu_{n} = ⟨ T_{n-1}, 1⟩.
```
Specialized construction and Cholesky factorization is given for this type.

See also [`GramMatrix`](@ref) for the general case.
"""
struct ChebyshevGramMatrix{T, V <: AbstractVector{T}} <: AbstractGramMatrix{T}
    μ::V
    n::Int
end

function ChebyshevGramMatrix(μ::V) where {T, V <: AbstractVector{T}}
    n = (length(μ)+1)÷2
    ChebyshevGramMatrix{T, V}(μ, n)
end

@inline size(G::ChebyshevGramMatrix) = (G.n, G.n)
@inline getindex(G::ChebyshevGramMatrix, i::Integer, j::Integer) = (G.μ[abs(i-j)+1] + G.μ[i+j-1])/2
@inline bandwidths(G::ChebyshevGramMatrix{T, <: PaddedVector{T}}) where T = (length(G.μ.args[2])-1, length(G.μ.args[2])-1)

#
# 2X'W-W*2X = G*J*G'
# This returns G, where J = [0 1; -1 0], respecting the skew-symmetry of the right-hand side.
# We use twice the Chebybshev Jacobi matrix so that subsequent arithmetic is easier.
#
function compute_skew_generators(W::ChebyshevGramMatrix{T}) where T
    μ = W.μ
    n = size(W, 1)
    G = zeros(T, n, 2)
    G[n, 1] = one(T)
    @inbounds @simd for j in 1:n-1
        G[j, 2] = -(μ[n+2-j] + μ[n+j])/2
    end
    G
end

function cholesky(W::ChebyshevGramMatrix{T}) where T
    n = size(W, 1)
    G = compute_skew_generators(W)
    L = zeros(T, n, n)
    c = W[:, 1]
    ĉ = zeros(T, n)
    l = zeros(T, n)
    v = zeros(T, n)
    row1 = zeros(T, n)
    fastcholesky!(L, G, c, ĉ, l, v, row1, n)
    return Cholesky(L, 'L', 0)
end

function fastcholesky!(L::Matrix{T}, G, c, ĉ, l, v, row1, n) where T
    @inbounds @simd for k in 1:n-1
        d = sqrt(c[k])
        for j in k:n
            L[j, k] = l[j] = c[j]/d
        end
        for j in k:n
            v[j] = G[j, 1]*G[k, 2] - G[j, 2]*G[k, 1]
        end
        if k == 1
            for j in 2:n-1
                ĉ[j] = (c[j+1] + c[j-1] + c[1]*row1[j] - row1[1]*c[j] - v[j])/2
            end
            ĉ[n] = (c[n-1] + c[1]*row1[n] - row1[1]*c[n] - v[n])/2
            cst = 2/d
        else
            for j in k+1:n-1
                ĉ[j] = c[j+1] + c[j-1] + c[k]*row1[j] - row1[k]*c[j] - v[j]
            end
            ĉ[n] = c[n-1] + c[k]*row1[n] - row1[k]*c[n] - v[n]
            cst = 1/d
        end
        for j in k+1:n
            row1[j] = -cst*l[j]
        end
        cst = c[k+1]/d
        for j in k:n
            c[j] = ĉ[j] - cst*l[j]
        end
        gd1 = G[k, 1]/d
        gd2 = G[k, 2]/d
        for j in k:n
            G[j, 1] -= l[j]*gd1
            G[j, 2] -= l[j]*gd2
        end
    end
    L[n, n] = sqrt(c[n])
end

function cholesky(W::ChebyshevGramMatrix{T, <: PaddedVector{T}}) where T
    n = size(W, 1)
    G = compute_skew_generators(W)
    L = BandedMatrix{T}(undef, (n, n), (bandwidth(W, 1), 0))
    c = W[:, 1]
    ĉ = zeros(T, n)
    l = zeros(T, n)
    v = zeros(T, n)
    row1 = zeros(T, n)
    fastcholesky!(L, G, c, ĉ, l, v, row1, n)
    return Cholesky(L, 'L', 0)
end

function fastcholesky!(L::BandedMatrix{T}, G, c, ĉ, l, v, row1, n) where T
    b = bandwidth(L, 1)
    @inbounds @simd for k in 1:n-1
        d = sqrt(c[k])
        for j in k:min(k+b, n)
            L[j, k] = l[j] = c[j]/d
        end
        for j in max(k, n-b-1):n
            v[j] = G[j, 1]*G[k, 2] - G[j, 2]*G[k, 1]
        end
        if k == 1
            for j in 2:min(b+2, n-1)
                ĉ[j] = (c[j+1] + c[j-1] + c[1]*row1[j] - row1[1]*c[j] - v[j])/2
            end
            if 1 ≥ n-b-1
                ĉ[n] = (c[n-1] + c[1]*row1[n] - row1[1]*c[n] - v[n])/2
            end
            cst = 2/d
        else
            for j in k+1:min(k+b+1, n-1)
                ĉ[j] = c[j+1] + c[j-1] + c[k]*row1[j] - row1[k]*c[j] - v[j]
            end
            if k ≥ n-b-1
                ĉ[n] = c[n-1] + c[k]*row1[n] - row1[k]*c[n] - v[n]
            end
            cst = 1/d
        end
        for j in k+1:min(k+b+1, n)
            row1[j] = -cst*l[j]
        end
        cst = c[k+1]/d
        for j in k:min(k+b+1, n)
            c[j] = ĉ[j] - cst*l[j]
        end
        gd1 = G[k, 1]/d
        gd2 = G[k, 2]/d
        for j in max(k, n-b-1):n
            G[j, 1] -= l[j]*gd1
            G[j, 2] -= l[j]*gd2
        end
    end
    L[n, n] = sqrt(c[n])
end
