"""
Bivariate modified moments of ``P_{n,k}(x,y) = p_{n-k}(x) q_k(y)`` with respect to the separable measure ``{\\rm d}\\mu(x, y) = {\\rm d}\\mu_1(x) {\\rm\\,d}\\mu_2(y)``:

```math
    \\iint_{\\mathbb{R}^2} P_{n,k}(x, y) {\\rm\\,d}\\mu(x, y) = \\int_{\\mathbb{R}} p_{n-k}(x) {\\rm\\,d}\\mu_1(x) \\int_{\\mathbb{R}} q_{k}(y) {\\rm\\,d}\\mu_2(y).
```
"""
function bivariatemoments(μ1::AbstractVector{T}, μ2::AbstractVector{T}) where T
    @assert length(μ1) == length(μ2)
    N = length(μ1)
    μ = BlockedVector{T}(undef, 1:N)
    for n in 0:N-1
        for k in 0:n
            μ[BlockIndex(n+1, k+1)] = μ1[n-k+1]*μ2[k+1]
        end
    end
    return μ
end

# These should live in BlockBandedMatrices.jl after PR #223

@inline function inbands_viewblock(A::BandedBlockBandedMatrix, KJ::Block{2})
    l,u = blockbandwidths(A)
    K,J = KJ.n
    BandedMatrices._BandedMatrix(view(A.data, Block(K-J+u+1, J)), length(axes(A,1)[Block(K)]), subblockbandwidths(A)...)
end

@inline function viewblock(A::BandedBlockBandedMatrix, KJ::Block{2})
    @boundscheck checkbounds(A, KJ)
    K,J = KJ.n
    if -A.u ≤ K-J ≤ A.l
        inbands_viewblock(A, KJ)
    else
        BandedMatrices._BandedMatrix(view(A.data, Block(1,1)), blocklengths(A,1)[Block(K)], (-40320,-40320))
    end
end

@inline function viewblock(A::AbstractMatrix, KJ::Block{2})
    view(A, KJ)
end



abstract type AbstractBivariateGramMatrix{T} <: AbstractBlockMatrix{T} end

@inline issymmetric(G::AbstractBivariateGramMatrix) = true
@inline isposdef(G::AbstractBivariateGramMatrix) = true


struct BivariateGramMatrix{T, WT <: AbstractMatrix{T}, XT <: AbstractMatrix{T}, YT <: AbstractMatrix{T}} <: AbstractBivariateGramMatrix{T}
    W::WT
    X::XT
    Y::YT
    function BivariateGramMatrix{T, WT, XT, YT}(W::WT, X::XT, Y::YT) where {T, WT, XT, YT}
        if size(W) ≠ size(X) ≠ size(Y)
            throw(ArgumentError("Cannot construct a BivariateGramMatrix with W, X, and Y of different sizes."))
        end
        if !issymmetric(W)
            throw(ArgumentError("Cannot construct a BivariateGramMatrix with a nonsymmetric W."))
        end
        if blockbandwidths(X) ≠ (1, 1)
            throw(ArgumentError("Cannot construct a BivariateGramMatrix with a nonblocktridiagonal X."))
        end
        if blockbandwidths(Y) ≠ (1, 1)
            throw(ArgumentError("Cannot construct a BivariateGramMatrix with a nonblocktridiagonal Y."))
        end
        new{T, WT, XT, YT}(W, X, Y)
    end
end

BivariateGramMatrix(W::WT, X::XT, Y::YT) where {T, WT <: AbstractMatrix{T}, XT <: AbstractMatrix{T}, YT <: AbstractMatrix{T}} = BivariateGramMatrix{T, WT, XT, YT}(W, X, Y)

@inline axes(G::BivariateGramMatrix) = axes(G.W)
@inline getindex(G::BivariateGramMatrix, i::Integer, j::Integer) = G.W[i, j]
@inline _blockindex_getindex(G::BivariateGramMatrix, bi::BlockIndex{2}) = _blockindex_getindex(G.W, bi)
@inline blockbandwidths(G::BivariateGramMatrix) = blockbandwidths(G.W)
@inline subblockbandwidths(G::BivariateGramMatrix) = subblockbandwidths(G.W)
@inline MemoryLayout(G::BivariateGramMatrix) = MemoryLayout(G.W)
@inline blockrowsupport(G::BivariateGramMatrix, j) = blockrowsupport(MemoryLayout(G), G.W, j)
@inline blockcolsupport(G::BivariateGramMatrix, j) = blockcolsupport(MemoryLayout(G), G.W, j)

BivariateGramMatrix(μ::AbstractBlockVector{T}, X::XT, Y::YT) where {T, XT <: AbstractMatrix{T}, YT <: AbstractMatrix{T}} = BivariateGramMatrix(μ, X, Y, one(T))
function BivariateGramMatrix(μ::AbstractBlockVector{T}, X::XT, Y::YT, p0::T) where {T, XT <: AbstractMatrix{T}, YT <: AbstractMatrix{T}}
    N = blocklength(μ)
    n = (N+1)÷2
    @assert N == blocksize(X, 1) == blocksize(X, 2) == blocksize(Y, 1) == blocksize(Y, 2)
    @assert blockbandwidths(X) == blockbandwidths(Y) == (1, 1)
    @assert subblockbandwidths(X) == (0, 0)
    @assert subblockbandwidths(Y) == (1, 1)
    W = BlockMatrix{T}(undef, 1:N, 1:N)
    if n > 0
        for m in 1:N
            W[Block(m, 1)] = p0*μ[Block(m, 1)]
        end
    end
    if n > 1
        for m in 2:N-1
            recurse!(W, X, Y, m)
        end
        symmetrize_block!(view(W, Block(2, 2)))
    end
    for n in 3:n
        for m in n:N-n+1
            recurse!(W, X, Y, m, n)
        end
        symmetrize_block!(view(W, Block(n, n)))
    end
    WN = BlockMatrix{T}(undef, 1:n, 1:n)
    for j in 1:n
        for k in j:n
            WN[Block(k, j)] = viewblock(W, Block(k, j))
        end
    end
    WS = Symmetric(WN, :L)
    XN = BandedBlockBandedMatrix{T}(undef, 1:n, 1:n, (1, 1), (0, 0))
    for j in 1:n
        for k in max(1, j-1):min(n, j+1)
            XN[Block(k, j)] = viewblock(X, Block(k, j))
        end
    end
    YN = BandedBlockBandedMatrix{T}(undef, 1:n, 1:n, (1, 1), (1, 1))
    for j in 1:n
        for k in max(1, j-1):min(n, j+1)
            YN[Block(k, j)] = viewblock(Y, Block(k, j))
        end
    end
    return BivariateGramMatrix(WS, XN, YN)
    #return BivariateGramMatrix(Symmetric(W[Block.(1:n), Block.(1:n)], :L), X[Block.(1:n), Block.(1:n)], Y[Block.(1:n), Block.(1:n)])
end

function BivariateGramMatrix(μ::BlockedVector{T, <: PaddedVector{T}}, X::XT, Y::YT, p0::T) where {T, XT <: AbstractMatrix{T}, YT <: AbstractMatrix{T}}
    N = blocklength(μ)
    b = blocklength(μ.args[2])-1
    n = (N+1)÷2
    @assert N == blocksize(X, 1) == blocksize(X, 2) == blocksize(Y, 1) == blocksize(Y, 2)
    @assert blockbandwidths(X) == blockbandwidths(Y) == (1, 1)
    @assert subblockbandwidths(X) == (0, 0)
    @assert subblockbandwidths(Y) == (1, 1)
    W = BandedBlockBandedMatrix{T}(undef, 1:N, 1:N, (b, 0), (b, b))
    if n > 0
        for m in 1:min(N, b+1)
            W[Block(m, 1)] = p0*μ[Block(m, 1)]
        end
    end
    if n > 1
        for m in 2:min(N-1, b+2)
            recurse!(W, X, Y, m)
        end
        symmetrize_block!(view(W, Block(2, 2)))
    end
    for n in 3:n
        for m in n:min(N-n+1, b+n)
            recurse!(W, X, Y, m, n)
        end
        symmetrize_block!(view(W, Block(n, n)))
    end
    WN = BandedBlockBandedMatrix{T}(undef, 1:n, 1:n, (b, 0), (b, b))
    for j in 1:n
        for k in j:min(n, j+b)
            WN[Block(k, j)] = viewblock(W, Block(k, j))
        end
    end
    WS = Symmetric(WN, :L)
    XN = BandedBlockBandedMatrix{T}(undef, 1:n, 1:n, (1, 1), (0, 0))
    for j in 1:n
        for k in max(1, j-1):min(n, j+1)
            XN[Block(k, j)] = viewblock(X, Block(k, j))
        end
    end
    YN = BandedBlockBandedMatrix{T}(undef, 1:n, 1:n, (1, 1), (1, 1))
    for j in 1:n
        for k in max(1, j-1):min(n, j+1)
            YN[Block(k, j)] = viewblock(Y, Block(k, j))
        end
    end
    return BivariateGramMatrix(WS, XN, YN)
    #return BivariateGramMatrix(Symmetric(W[Block.(1:n), Block.(1:n)], :L), X[Block.(1:n), Block.(1:n)], Y[Block.(1:n), Block.(1:n)])
end

function symmetrize_block!(W)
    @assert size(W, 1) == size(W, 2)
    @inbounds for j in 2:size(W, 2)
        r = colrange(W, j)
        ir = first(r):j-1
        for i in ir
            W[i, j] = W[j, i]
        end
    end
end

# n == 2
function recurse!(W, X, Y, m)
    # XW = X[Block(m-1, m)]'W[Block(m-1, 1)] + X[Block(m, m)]'W[Block(m, 1)] + X[Block(m+1, m)]'W[Block(m+1, 1)] - W[Block(m, 1)]*X[Block(1, 1)]
    # Xn = Diagonal(X[Block(2, 1)])
    # Wmn = zeros(T, m, 2)
    # Wmn[:, 1] .= XW/Xn
    Wa = view(W, Block(m-1, 1))
    Wb = view(W, Block(m, 1))
    Wc = view(W, Block(m+1, 1))
    We = view(viewblock(W, Block(m, 2)), 1:m, 1:1)

    Xa = view(X.data, Block(1, m))
    Xb = view(X.data, Block(2, m))
    Xc = view(X.data, Block(3, m))
    Xd = view(X.data, Block(2, 1))
    Xn = view(X.data, Block(3, 1))

    r = colrange(Wa, 1)
    jr = first(r):min(last(r), m-1)
    @inbounds for j in jr
        We[j, 1] = Xa[j]*Wa[j, 1]
    end
    We[m, 1] = 0
    @inbounds for j in colrange(Wb, 1)
        We[j, 1] += Xb[j]*Wb[j, 1]
    end
    r = colrange(Wc, 1)
    jr = first(r):min(last(r), m)
    @inbounds for j in jr
        We[j, 1] += Xc[j]*Wc[j, 1]
    end
    @inbounds for j in colrange(Wb, 1)
        We[j, 1] -= Wb[j, 1]*Xd[1]
    end
    @inbounds for j in colrange(We, 1)
        We[j, 1] /= Xn[1]
    end

    # YWnm1 = Y[Block(m-1, m)]'view(Wa, :, 1) + Y[Block(m, m)]'view(Wb, :, 1) + Y[Block(m+1, m)]'view(Wc, :, 1) - Wb*view(Y[Block(1, 1)], :, 1)
    # Yn = Y[Block(2, 1)]
    # Wmn[:, 2] .= (YWnm1 .- Yn[1, 1].*Wmn[:, 1])./Yn[2, 1]
    Ya = viewblock(Y, Block(m-1, m))
    Yb = viewblock(Y, Block(m, m))
    Yc = viewblock(Y, Block(m+1, m))
    Yd = viewblock(Y, Block(1, 1))

    We = view(viewblock(W, Block(m, 2)), 1:m, 2)
    mul!(We, Ya', view(Wa, :, 1))
    mul!(We, Yb', view(Wb, :, 1), 1, 1)
    mul!(We, Yc', view(Wc, :, 1), 1, 1)
    mul!(We, Wb, view(Yd, :, 1), -1, 1)
    Yn = viewblock(Y, Block(2, 1))
    We .-= Yn[1, 1].*view(viewblock(W, Block(m, 2)), 1:m, 1)
    rdiv!(We, Yn[2, 1])
end

function recurse!(W, X, Y, m, n)
    # XW = X[Block(m-1, m)]'W[Block(m-1, n-1)] + X[Block(m, m)]'W[Block(m, n-1)] + X[Block(m+1, m)]'W[Block(m+1, n-1)] - W[Block(m, n-1)]*X[Block(n-1, n-1)] - W[Block(m, n-2)]*X[Block(n-2, n-1)]
    # Xn = Diagonal(X[Block(n, n-1)])
    # Wmn = zeros(T, m, n)
    # Wmn[:, 1:n-1] .= XW/Xn
    Wa = view(W, Block(m-1, n-1))
    Wb = view(W, Block(m, n-1))
    Wc = view(W, Block(m+1, n-1))
    Wd = view(W, Block(m, n-2))
    We = view(viewblock(W, Block(m, n)), 1:m, 1:n-1)

    Xa = view(X.data, Block(1, m))
    Xb = view(X.data, Block(2, m))
    Xc = view(X.data, Block(3, m))
    Xd = view(X.data, Block(2, n-1))
    Xe = view(X.data, Block(1, n-1))
    Xn = view(X.data, Block(3, n-1))

    @inbounds for k in 1:n-1
        r = colrange(Wa, k)
        jr = first(r):min(last(r), m-1)
        for j in jr
            We[j, k] = Xa[j]*Wa[j, k]
        end
        We[m, k] = 0
    end
    @inbounds for k in 1:n-1
        for j in colrange(Wb, k)
            We[j, k] += Xb[j]*Wb[j, k]
        end
    end
    @inbounds for k in 1:n-1
        r = colrange(Wc, k)
        jr = first(r):min(last(r), m)
        for j in jr
            We[j, k] += Xc[j]*Wc[j, k]
        end
    end
    @inbounds for k in 1:n-1
        for j in colrange(Wb, k)
            We[j, k] -= Wb[j, k]*Xd[k]
        end
    end
    @inbounds for k in 1:n-2
        for j in colrange(Wd, k)
            We[j, k] -= Wd[j, k]*Xe[k]
        end
    end
    @inbounds for k in 1:n-1
        for j in colrange(We, k)
            We[j, k] /= Xn[k]
        end
    end

    # YWnm1 = Y[Block(m-1, m)]'view(Wa, :, n-1) + Y[Block(m, m)]'view(Wb, :, n-1) + Y[Block(m+1, m)]'view(Wc, :, n-1) - Wb*view(Y[Block(n-1, n-1)], :, n-1) - Wd*view(Y[Block(n-2, n-1)], :, n-1)
    # Yn = Y[Block(n, n-1)]
    # Wmn[:, n] .= (YWnm1 .- Yn[n-2, n-1].*Wmn[:, n-2] .- Yn[n-1, n-1].*Wmn[:, n-1])./Yn[n, n-1]
    Ya = viewblock(Y, Block(m-1, m))
    Yb = viewblock(Y, Block(m, m))
    Yc = viewblock(Y, Block(m+1, m))
    Yd = viewblock(Y, Block(n-1, n-1))
    Ye = viewblock(Y, Block(n-2, n-1))

    We = view(viewblock(W, Block(m, n)), 1:m, n)
    mul!(We, Ya', view(Wa, :, n-1))
    mul!(We, Yb', view(Wb, :, n-1), 1, 1)
    mul!(We, Yc', view(Wc, :, n-1), 1, 1)
    mul!(We, Wb, view(Yd, :, n-1), -1, 1)
    mul!(We, Wd, view(Ye, :, n-1), -1, 1)
    Yn = viewblock(Y, Block(n, n-1))
    We .-= Yn[n-2, n-1].*view(viewblock(W, Block(m, n)), 1:m, n-2) .+ Yn[n-1, n-1].*view(viewblock(W, Block(m, n)), 1:m, n-1)
    rdiv!(We, Yn[n, n-1])
end



compute_skew_generators(::Val{1}, G::BivariateGramMatrix) = compute_skew_generators(G.X, G)
compute_skew_generators(::Val{2}, G::BivariateGramMatrix) = compute_skew_generators(G.Y, G)
#
# The computation of the skew-symmetric generators for the Gram matrix equation:
# Z'W - WZ = GJGᵀ,
# where Z = X or Y is a bivariate Jacobi matrix that is block tridiagonal.
# We know that the left-hand side has zeros in the first (n-1) x (n-1) blocks.
# Thus, determining G can be done with the matrix-matrix product in the last column.
# This determines the first (n-1) blocks of the generators. To find the last,
# we observe that the skew-symmetry in the last block leaves it underdetermined,
# but it can be chosen to be strictly upper triangular.
#
function compute_skew_generators(Z::AbstractMatrix{T}, W::BivariateGramMatrix{T}) where T
    n = blocksize(W, 1)
    G = BlockMatrix{T}(undef_blocks, 1:n, SVector(n, n))
    for j in 1:n-1
        G.blocks[j, 1] = zeros(T, j, n)
    end
    G.blocks[n, 1] = Matrix{T}(I, n, n)
    v = W[Block.(1:n), Block(n-1)]*Z[Block(n-1), Block(n)] + W[Block.(1:n), Block(n)]*Z[Block(n), Block(n)] - Z'W[Block.(1:n), Block(n)]
    for j in 1:n-1
        G.blocks[j, 2] = v[Block(j)]
    end
    G.blocks[n, 2] = triu!(v[Block(n)])
    return G
end





struct BivariateChebyshevGramMatrix{T, BV <: AbstractBlockVector{T}, BS <: NTuple{2, AbstractUnitRange{Int}}} <: AbstractBivariateGramMatrix{T}
    μ::BV
    axes::BS
end

function BivariateChebyshevGramMatrix(μ::AbstractBlockVector{T}) where T
    n = (blocklength(μ)+1)÷2
    BivariateChebyshevGramMatrix(μ, (blockedrange(1:n), blockedrange(1:n)))
end

@inline axes(G::BivariateChebyshevGramMatrix) = G.axes

Base.@propagate_inbounds getindex(G::BivariateChebyshevGramMatrix{T}, blockindex::BlockIndex{2}) where T = _blockindex_getindex(G, blockindex)

@inline function _blockindex_getindex(G::BivariateChebyshevGramMatrix{T}, bi::BlockIndex{2}) where T
    @boundscheck blockcheckbounds(G, Block(bi.I))
    m, n = bi.I
    j, k = bi.α
    @boundscheck (1 ≤ j ≤ m) && (1 ≤ k ≤ n)
    μ = G.μ
    v = (μ[BlockIndex(m+n-1, j+k-1)]+μ[BlockIndex(abs(m-j-n+k)+j+k-1, j+k-1)]+μ[BlockIndex(m-j+n-k+abs(j-k)+1, abs(j-k)+1)]+μ[BlockIndex(abs(m-j-n+k)+abs(j-k)+1, abs(j-k)+1)])/4
    return v
end

@inline function getindex(G::BivariateChebyshevGramMatrix{T}, i::Vararg{Integer, 2}) where T
    @boundscheck checkbounds(G, i...)
    @inbounds v = G[findblockindex.(axes(G), i)...]
    return v
end

@inline function blockbandwidths(G::BivariateChebyshevGramMatrix{T, <: BlockedVector{T, <: PaddedVector{T}}}) where T
    N = length(G.μ.blocks.args[2])
    b = ceil(Int, (-1+sqrt(1+8N))/2) - 1
    return (b, b)
end
@inline function subblockbandwidths(G::BivariateChebyshevGramMatrix{T, <: BlockedVector{T, <: PaddedVector{T}}}) where T
    N = length(G.μ.blocks.args[2])
    b = ceil(Int, (-1+sqrt(1+8N))/2) - 1
    return (b, b)
end
@inline MemoryLayout(G::BivariateChebyshevGramMatrix{T, <: BlockedVector{T, <: PaddedVector{T}}}) where T = BandedBlockBandedLayout()

#
# The computation of the skew-symmetric generators for the bivariate Chebyshev--Gram matrix equation:
# Z'W-WZ = GJGᵀ,
# where Z = X or Y is a bivariate Chebyshev Jacobi matrix that is block tridiagonal.
# Since the bivariate Chebyhsev-Gram matrix is special, these formulas compute G
# without the use of X or Y (thus without matrix multiplication).
#
function compute_skew_generators(::Val{1}, W::BivariateChebyshevGramMatrix{T}) where T
    N = blocksize(W, 1)
    G = BlockMatrix{T}(undef_blocks, 1:N, SVector(N, N))
    for j in 1:N-1
        G.blocks[j, 1] = zeros(T, j, N)
    end
    G.blocks[N, 1] = Matrix{T}(I, N, N)
    μ = W.μ
    @inbounds for m in 1:N-1
        GB = zeros(T, m, N)
        for k in 1:N-1
            for j in 1:m
                GB[j, k] = -(μ[BlockIndex(m+N, j+k-1)]+μ[BlockIndex(abs(m-j-N-1+k)+j+k-1, j+k-1)]+μ[BlockIndex(m-j+N+1-k+abs(j-k)+1, abs(j-k)+1)]+μ[BlockIndex(abs(m-j-N-1+k)+abs(j-k)+1, abs(j-k)+1)])/8
            end
        end
        for j in 1:m
            GB[j, N] = -(μ[BlockIndex(m+N, j+N-1)]+μ[BlockIndex(abs(m-j-1)+j+N-1, j+N-1)]+μ[BlockIndex(m-j+1+abs(j-N)+1, abs(j-N)+1)]+μ[BlockIndex(abs(m-j-1)+abs(j-N)+1, abs(j-N)+1)])/4
        end
        G.blocks[m, 2] = GB
    end
    GB = zeros(T, N, N)
    @inbounds for k in 1:N-1
        for j in 1:N
            if abs(k-j-1)+j+k-1 < 2N
                GB[j, k] -= μ[BlockIndex(abs(k-j-1)+j+k-1, j+k-1)]
            end
            if 2N-j+1-k+abs(j-k)+1 < 2N
                GB[j, k] -= μ[BlockIndex(2N-j+1-k+abs(j-k)+1, abs(j-k)+1)]
            end
            if abs(k-j-1)+abs(j-k)+1 < 2N
                GB[j, k] -= μ[BlockIndex(abs(k-j-1)+abs(j-k)+1, abs(j-k)+1)]
            end
            GB[j, k] /= 8
        end
    end
    @inbounds for j in 1:N
        if abs(N-j-1)+j+N-1 < 2N
            GB[j, N] -= μ[BlockIndex(abs(N-j-1)+j+N-1, j+N-1)]
        end
        if N-j+1+abs(j-N)+1 < 2N
            GB[j, N] -= μ[BlockIndex(N-j+1+abs(j-N)+1, abs(j-N)+1)]
        end
        if abs(N-j-1)+abs(j-N)+1 < 2N
            GB[j, N] -= μ[BlockIndex(abs(N-j-1)+abs(j-N)+1, abs(j-N)+1)]
        end
        GB[j, N] /= 4
    end
    @inbounds for k in 1:N
        for j in 1:k
            GB[j, k] -= GB[k, j]
            GB[k, j] = zero(T)
        end
    end
    G.blocks[N, 2] = GB

    G
end

function compute_skew_generators(::Val{2}, W::BivariateChebyshevGramMatrix{T}) where T
    N = blocksize(W, 1)
    G = BlockMatrix{T}(undef_blocks, 1:N, SVector(N, N))
    for j in 1:N-1
        G.blocks[j, 1] = zeros(T, j, N)
    end
    G.blocks[N, 1] = Matrix{T}(I, N, N)
    μ = W.μ
    @inbounds for m in 1:N-1
        GB = zeros(T, m, N)
        for j in 1:m
            GB[j, 1] = -(μ[BlockIndex(m+N, j+1)]+μ[BlockIndex(abs(m-j-N+1)+j+1, j+1)]+μ[BlockIndex(m-j+N-1+abs(j-2)+1, abs(j-2)+1)]+μ[BlockIndex(abs(m-j-N+1)+abs(j-2)+1, abs(j-2)+1)])/4
        end
        for k in 2:N
            for j in 1:m
                GB[j, k] = -(μ[BlockIndex(m+N, j+k)]+μ[BlockIndex(abs(m-j-N+k)+j+k, j+k)]+μ[BlockIndex(m-j+N-k+abs(j-k-1)+1, abs(j-k-1)+1)]+μ[BlockIndex(abs(m-j-N+k)+abs(j-k-1)+1, abs(j-k-1)+1)])/8
            end
        end
        G.blocks[m, 2] = GB
    end
    GB = zeros(T, N, N)
    @inbounds for j in 1:N
        if abs(1-j)+j+1 < 2N
            GB[j, 1] -= μ[BlockIndex(abs(1-j)+j+1, j+1)]
        end
        if 2N-j-1+abs(j-2)+1 < 2N
            GB[j, 1] -= μ[BlockIndex(2N-j-1+abs(j-2)+1, abs(j-2)+1)]
        end
        if abs(1-j)+abs(j-2)+1 < 2N
            GB[j, 1] -= μ[BlockIndex(abs(1-j)+abs(j-2)+1, abs(j-2)+1)]
        end
        GB[j, 1] /= 4
    end
    @inbounds for k in 2:N
        for j in 1:N
            if abs(k-j)+j+k < 2N
                GB[j, k] -= μ[BlockIndex(abs(k-j)+j+k, j+k)]
            end
            if 2N-j-k+abs(j-k-1)+1 < 2N
                GB[j, k] -= μ[BlockIndex(2N-j-k+abs(j-k-1)+1, abs(j-k-1)+1)]
            end
            if abs(k-j)+abs(j-k-1)+1 < 2N
                GB[j, k] -= μ[BlockIndex(abs(k-j)+abs(j-k-1)+1, abs(j-k-1)+1)]
            end
            GB[j, k] /= 8
        end
    end
    @inbounds for k in 1:N
        for j in 1:k
            GB[j, k] -= GB[k, j]
            GB[k, j] = zero(T)
        end
    end
    G.blocks[N, 2] = GB

    G
end



## Move to tests?

function _chebyshev_x(::Type{T}, n::Integer) where T
    X = BandedBlockBandedMatrix{T}(undef, 1:n, 1:n, (1, 1), (0, 0))
    dat = X.data.blocks
    @inbounds for j in 1:n*(n+1)÷2
        dat[2, j] = zero(T)
    end
    @inbounds for j in 1:n-1
        for k in 1:j-1
            dat[3, k+j*(j-1)÷2] = one(T)/2
        end
        dat[3, j+j*(j-1)÷2] = one(T)
    end
    @inbounds for j in 2:n
        for k in 1:j
            dat[1, k+j*(j-1)÷2] = one(T)/2
        end
    end

    return X
end

function _chebyshev_y(::Type{T}, n::Integer) where T
    Y = BandedBlockBandedMatrix{T}(undef, 1:n, 1:n, (1, 1), (1, 1))
    dat = Y.data.blocks
    @inbounds for k in 2:8
        for j in 1:n*(n+1)÷2
            dat[k, j] = zero(T)
        end
    end
    @inbounds for j in 1:n-1
        dat[9, 1+j*(j-1)÷2] = one(T)
        for k in 2:j
            dat[9, k+j*(j-1)÷2] = one(T)/2
        end
    end
    @inbounds for j in 2:n
        for k in 2:j
            dat[1, k+j*(j-1)÷2] = one(T)/2
        end
    end

    return Y
end
