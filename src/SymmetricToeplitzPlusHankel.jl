struct SymmetricToeplitzPlusHankel{T} <: AbstractMatrix{T}
    v::Vector{T}
    n::Int
end

function SymmetricToeplitzPlusHankel(v::Vector{T}) where T
    n = (length(v)+1)÷2
    SymmetricToeplitzPlusHankel{T}(v, n)
end

size(A::SymmetricToeplitzPlusHankel{T}) where T = (A.n, A.n)
getindex(A::SymmetricToeplitzPlusHankel{T}, i::Integer, j::Integer) where T = A.v[abs(i-j)+1] + A.v[i+j-1]

struct SymmetricBandedToeplitzPlusHankel{T} <: BandedMatrices.AbstractBandedMatrix{T}
    v::Vector{T}
    n::Int
    b::Int
end

function SymmetricBandedToeplitzPlusHankel(v::Vector{T}, n::Integer) where T
    SymmetricBandedToeplitzPlusHankel{T}(v, n, length(v)-1)
end

size(A::SymmetricBandedToeplitzPlusHankel{T}) where T = (A.n, A.n)
function getindex(A::SymmetricBandedToeplitzPlusHankel{T}, i::Integer, j::Integer) where T
    v = A.v
    if abs(i-j) < length(v)
        if i+j-1 ≤ length(v)
            v[abs(i-j)+1] + v[i+j-1]
        else
            v[abs(i-j)+1]
        end
    else
        zero(T)
    end
end
bandwidths(A::SymmetricBandedToeplitzPlusHankel{T}) where T = (A.b, A.b)

#
# X'W-W*X = G*J*G'
# This returns G and J, where J = [0 I; -I 0], respecting the skew-symmetry of the right-hand side.
#
function compute_skew_generators(A::SymmetricToeplitzPlusHankel{T}) where T
    v = A.v
    n = size(A, 1)
    J = [zero(T) one(T); -one(T) zero(T)]
    G = zeros(T, n, 2)
    G[n, 1] = one(T)
    u2 = reverse(v[2:n+1])
    u2[1:n-1] .+= v[n+1:2n-1]
    G[:, 2] .= -u2
    G, J
end

function cholesky(A::SymmetricToeplitzPlusHankel{T}) where T
    n = size(A, 1)
    G, J = compute_skew_generators(A)
    L = zeros(T, n, n)
    c = A[:, 1]
    ĉ = zeros(T, n)
    l = zeros(T, n)
    v = zeros(T, n)
    row1 = zeros(T, n)
    STPHcholesky!(L, G, c, ĉ, l, v, row1, n)
    return Cholesky(L, 'L', 0)
end

function STPHcholesky!(L::Matrix{T}, G, c, ĉ, l, v, row1, n) where T
    @inbounds @simd for k in 1:n-1
        d = sqrt(c[1])
        for j in 1:n-k+1
            L[j+k-1, k] = l[j] = c[j]/d
        end
        for j in 1:n-k+1
            v[j] = G[j, 1]*G[1, 2] - G[j, 2]*G[1, 1]
        end
        X21 = k == 1 ? T(2) : T(1)
        ĉ[1] = (c[2] - v[1])/X21
        for j in 2:n-k
            ĉ[j] = (c[j+1] + c[j-1] + c[1]*row1[j] - row1[1]*c[j] - v[j])/X21
        end
        ĉ[n-k+1] = (c[n-k] + c[1]*row1[n-k+1] - row1[1]*c[n-k+1] - v[n-k+1])/X21
        cst = c[2]/d
        for j in 1:n-k
            c[j] = ĉ[j+1] - cst*l[j+1]
        end
        cst = X21/d
        for j in 1:n-k
            row1[j] = -cst*l[j+1]
        end
        gd1 = G[1, 1]/d
        gd2 = G[1, 2]/d
        for j in 1:n-k
            G[j, 1] = G[j+1, 1] - l[j+1]*gd1
            G[j, 2] = G[j+1, 2] - l[j+1]*gd2
        end
    end
    L[n, n] = sqrt(c[1])
end

function cholesky(A::SymmetricBandedToeplitzPlusHankel{T}) where T
    n = size(A, 1)
    b = A.b
    R = BandedMatrix{T}(undef, (n, n), (0, bandwidth(A, 2)))
    c = A[1:b+2, 1]
    ĉ = zeros(T, b+3)
    l = zeros(T, b+3)
    row1 = zeros(T, b+2)
    SBTPHcholesky!(R, c, ĉ, l, row1, n, b)
    return Cholesky(R, 'U', 0)
end

function SBTPHcholesky!(R::BandedMatrix{T}, c, ĉ, l, row1, n, b) where T
    @inbounds @simd for k in 1:n
        d = sqrt(c[1])
        for j in 1:b+1
            l[j] = c[j]/d
        end
        for j in 1:min(n-k+1, b+1)
            R[k, j+k-1] = l[j]
        end
        X21 = k == 1 ? T(2) : T(1)
        ĉ[1] = c[2]/X21
        for j in 2:b+1
            ĉ[j] = (c[j+1] + c[j-1] + c[1]*row1[j] - row1[1]*c[j])/X21
        end
        ĉ[b+2] = (c[b+1] + c[1]*row1[b+2] - row1[1]*c[b+2])/X21
        cst = c[2]/d
        for j in 1:b+2
            c[j] = ĉ[j+1] - cst*l[j+1]
        end
        cst = X21/d
        for j in 1:b+2
            row1[j] = -cst*l[j+1]
        end
    end
end
