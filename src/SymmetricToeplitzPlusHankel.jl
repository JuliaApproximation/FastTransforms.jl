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
# Jac*W-W*Jac' = G*J*G'
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
    r = A[:, 1]
    r2 = zeros(T, n)
    l = zeros(T, n)
    v = zeros(T, n)
    col1 = zeros(T, n)
    STPHcholesky!(L, G, r, r2, l, v, col1, n)
    return Cholesky(L, 'L', 0)
end

function STPHcholesky!(L::Matrix{T}, G, r, r2, l, v, col1, n) where T
    @inbounds @simd for k in 1:n-1
        x = sqrt(r[1])
        for j in 1:n-k+1
            L[j+k-1, k] = l[j] = r[j]/x
        end
        for j in 1:n-k+1
            v[j] = G[j, 1]*G[1,2]-G[j,2]*G[1,1]
        end
        F12 = k == 1 ? T(2) : T(1)
        r2[1] = (r[2] - v[1])/F12
        for j in 2:n-k
            r2[j] = (r[j+1]+r[j-1] + r[1]*col1[j] - col1[1]*r[j] - v[j])/F12
        end
        r2[n-k+1] = (r[n-k] + r[1]*col1[n-k+1] - col1[1]*r[n-k+1] - v[n-k+1])/F12
        cst = r[2]/x
        for j in 1:n-k
            r[j] = r2[j+1] - cst*l[j+1]
        end
        for j in 1:n-k
            col1[j] = -F12/x*l[j+1]
        end
        c1 = G[1, 1]
        c2 = G[1, 2]
        for j in 1:n-k
            G[j, 1] = G[j+1, 1] - l[j+1]*c1/x
            G[j, 2] = G[j+1, 2] - l[j+1]*c2/x
        end
    end
    L[n, n] = sqrt(r[1])
end

function cholesky(A::SymmetricBandedToeplitzPlusHankel{T}) where T
    n = size(A, 1)
    b = A.b
    R = BandedMatrix{T}(undef, (n, n), (0, bandwidth(A, 2)))
    r = A[1:b+2, 1]
    r2 = zeros(T, b+3)
    l = zeros(T, b+3)
    col1 = zeros(T, b+2)
    SBTPHcholesky!(R, r, r2, l, col1, n, b)
    return Cholesky(R, 'U', 0)
end

function SBTPHcholesky!(R::BandedMatrix{T}, r, r2, l, col1, n, b) where T
    @inbounds @simd for k in 1:n
        x = sqrt(r[1])
        for j in 1:b+1
            l[j] = r[j]/x
        end
        for j in 1:min(n-k+1, b+1)
            R[k, j+k-1] = l[j]
        end
        F12 = k == 1 ? T(2) : T(1)
        r2[1] = r[2]/F12
        for j in 2:b+1
            r2[j] = (r[j+1]+r[j-1] + r[1]*col1[j] - col1[1]*r[j])/F12
        end
        r2[b+2] = (r[b+1] + r[1]*col1[b+2] - col1[1]*r[b+2])/F12
        cst = r[2]/x
        for j in 1:b+2
            r[j] = r2[j+1] - cst*l[j+1]
        end
        for j in 1:b+2
            col1[j] = -F12/x*l[j+1]
        end
    end
end
