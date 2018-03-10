function sph_zero_spurious_modes!(A::AbstractMatrix)
    M, N = size(A)
    n = N÷2
    @inbounds for j = 1:n-1
        @simd for i = M-j+1:M
            A[i,2j] = 0
            A[i,2j+1] = 0
        end
    end
    @inbounds @simd for i = M-n+1:M
        A[i,2n] = 0
        2n < N && (A[i,2n+1] = 0)
    end
    A
end

function sphrand(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = rand(T)
    end
    for j = 1:n-1
        for i = 1:m-j
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function sphrandn(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = randn(T)
    end
    for j = 1:n-1
        for i = 1:m-j
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function sphones(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = one(T)
    end
    for j = 1:n-1
        for i = 1:m-j
            A[i,2j] = one(T)
            A[i,2j+1] = one(T)
        end
    end
    A
end

sphzeros(::Type{T}, m::Int, n::Int) where T = zeros(T, m, 2n-1)

function normalizecolumns!(A::AbstractMatrix)
    m, n = size(A)
    @inbounds for j = 1:n
        nrm = zero(eltype(A))
        for i = 1:m
            nrm += abs2(A[i,j])
        end
        nrm = sqrt(nrm)
        for i = 1:m
            A[i,j] /= nrm
        end
    end
    A
end

function maxcolnorm(A::AbstractMatrix)
    m, n = size(A)
    nrm = zeros(n)
    @inbounds for j = 1:n
        nrm[j] = 0
        for i = 1:m
            nrm[j] += abs2(A[i,j])
        end
        nrm[j] = sqrt(nrm[j])
    end
    norm(nrm, Inf)
end

doc"""
Pointwise evaluation of spherical harmonic:

```math
Y_\ell^m(\theta,\varphi) = \frac{e^{{\rm i} m\varphi}}{\sqrt{2\pi}} {\rm i}^{m+|m|}\sqrt{(\ell+\frac{1}{2})\frac{(\ell-m)!}{(\ell+m)!}} P_\ell^m(\cos\theta).
```
"""
sphevaluate(θ, φ, L, M) = sphevaluatepi(θ/π, φ/π, L, M)

sphevaluatepi(θ::Number, φ::Number, L::Integer, M::Integer) = sphevaluatepi(θ,L,M)*sphevaluatepi(φ,M)

function sphevaluatepi(θ::Number, L::Integer, M::Integer)
    ret = one(θ)/sqrt(two(θ))
    if M < 0 M = -M end
    c, s = cospi(θ), sinpi(θ)
    for m = 1:M
        ret *= sqrt((m+half(θ))/m)*s
    end
    tc = two(c)*c

    if L == M
        return ret
    elseif L == M+1
        return sqrt(two(θ)*M+3)*c*ret
    else
        temp = ret
        ret *= sqrt(two(θ)*M+3)*c
        for l = M+1:L-1
            ret, temp = (sqrt(l+half(θ))*tc*ret - sqrt((l-M)*(l+M)/(l-half(θ)))*temp)/sqrt((l-M+1)*(l+M+1)/(l+3half(θ))), ret
        end
        return ret
    end
end

sphevaluatepi(φ::Number, M::Integer) = complex(cospi(M*φ),sinpi(M*φ))/sqrt(two(φ)*π)
