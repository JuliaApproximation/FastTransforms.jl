abstract type AbstractGramMatrix{T} <: LayoutMatrix{T} end

@inline issymmetric(G::AbstractGramMatrix) = true
@inline isposdef(G::AbstractGramMatrix) = true
@inline colsupport(G::AbstractGramMatrix, j) = colrange(G, j)

"""
    GramMatrix(W::AbstractMatrix, X::AbstractMatrix)

Construct a symmetric positive-definite Gram matrix with data stored in ``W``.
Given a family of orthogonal polynomials ``{\\bf P}(x) = \\{p_0(x), p_1(x),\\ldots\\}``
and a continuous inner product ``\\langle f, g\\rangle``, the Gram matrix is defined by:
```math
W_{i,j} = \\langle p_{i-1}, p_{j-1}\\rangle.
```
Moreover, given ``X``, the transposed Jacobi matrix that satisfies ``x {\\bf P}(x) = {\\bf P}(x) X``,
the Gram matrix satisfies the skew-symmetric rank-2 displacement equation (``X = X_{1:n, 1:n}``):
```math
X^\\top W - WX = GJG^\\top,
```
where ``J = \\begin{pmatrix} 0 & 1\\\\ -1 & 0\\end{pmatrix}`` and where:
```math
G_{:, 1} = e_n,\\quad{\\rm and}\\quad G_{:, 2} = W_{n-1, :}X_{n-1, n} - X^\\top W_{:, n}.
```
Fast (``O(n^2)``) Cholesky factorization of the Gram matrix returns the
connection coefficients between ``{\\bf P}(x)`` and the polynomials ``{\\bf Q}(x)``
orthogonal in the modified inner product, ``{\\bf P}(x) = {\\bf Q}(x) R``.
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

#
# X'W-W*X = G*J*G'
# This returns G, where J = [0 1; -1 0], respecting the skew-symmetry of the right-hand side.
#
function compute_skew_generators(W::GramMatrix{T}) where T
    X = W.X
    n = size(W, 1)
    G = zeros(T, n, 2)
    G[n, 1] = one(T)
    G[:, 2] .= W[n-1, :]*X[n-1, n] + W[n, :]*X[n, n] - X'W[:, n]
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
\\mu_{n} = \\langle T_{n-1}, 1\\rangle.
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
    G[n, 2] = -μ[2]/2
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
