struct ToeplitzPlusHankel{T, S, P1 <: Plan{S}, P2 <: Plan{S}} <: AbstractMatrix{T}
    tc::Vector{T}
    tr::Vector{T}
    h::Vector{T}
    th_dft::Matrix{S}
    tht_dft::Matrix{S}
    temp::Matrix{S}
    plan::P1
    iplan::P2
    size::NTuple{2, Int}
end

# enforces tr[1] == tc[1]
function ToeplitzPlusHankel(tc::Vector{T}, tr::Vector{T}, h::Vector{T}) where T
    m = length(tc)
    n = length(tr)
    @assert length(h) == m+n-1
    tr[1] = tc[1]
    mn = m+n
    S = promote_type(float(T), Complex{Float32})
    th_dft = Matrix{S}(undef, mn, 2)
    copyto!(th_dft, 1, tc, 1, m)
    th_dft[m+1, 1] = zero(T)
    copyto!(th_dft, m+2, Iterators.reverse(tr), 1, n-1)
    copyto!(th_dft, mn+1, h, n, m)
    th_dft[m+1, 2] = zero(T)
    copyto!(th_dft, mn+m+2, h, 1, n-1)
    tht_dft = Matrix{S}(undef, mn, 2)
    copyto!(tht_dft, 1, tr, 1, n)
    tht_dft[n+1, 1] = zero(T)
    copyto!(tht_dft, n+2, Iterators.reverse(tc), 1, m-1)
    copyto!(tht_dft, mn+1, h, m, n)
    tht_dft[n+1, 2] = zero(T)
    copyto!(tht_dft, mn+n+2, h, 1, m-1)

    plan = plan_fft!(th_dft, 1)
    plan*th_dft
    plan*tht_dft
    temp = zeros(S, mn, 2)
    iplan = inv(plan)

    ToeplitzPlusHankel{T, S, typeof(plan), typeof(iplan)}(tc, tr, h, th_dft, tht_dft, temp, plan, iplan, (m, n))
end

# A ChebyshevGramMatrix isa (symmetric positive-definite) ToeplitzPlusHankel matrix.
function ToeplitzPlusHankel(G::ChebyshevGramMatrix)
    n = size(G, 1)
    ToeplitzPlusHankel(G.μ[1:n]/2, G.μ[1:n]/2, G.μ/2)
end

size(A::ToeplitzPlusHankel) = A.size
getindex(A::ToeplitzPlusHankel, i::Integer, j::Integer) = (i ≥ j ? A.tc[i-j+1] : A.tr[j-i+1]) + A.h[i+j-1]

# A view of a T+H is also T+H.
function getindex(A::ToeplitzPlusHankel, ir::UnitRange{Int}, jr::UnitRange{Int})
    fir, lir = first(ir), last(ir)
    fjr, ljr = first(jr), last(jr)
    if fir ≥ fjr
        tc = A.tc[fir-fjr+1:lir-fjr+1]
        tr = [A.tc[fir-fjr+1:-1:max(1, fir-ljr+1)]; A.tr[2:ljr-fir+1]]
    else
        tc = [A.tr[fjr-fir+1:-1:max(1, fjr-lir+1)]; A.tc[2:lir-fjr+1]]
        tr = A.tr[fjr-fir+1:ljr-fir+1]
    end
    ToeplitzPlusHankel(tc, tr, A.h[fir+fjr-1:lir+ljr-1])
end


# y ← A x α + y β
function mul!(y::StridedVector{T}, A::ToeplitzPlusHankel{T}, x::StridedVector{T}, α::S, β::S) where {T <: Real, S <: Real}
    m, n = size(A)
    @assert m == length(y)
    @assert n == length(x)
    mn = m+n
    th_dft = A.th_dft
    temp = A.temp
    plan = A.plan
    iplan = A.iplan

    copyto!(temp, 1, x, 1, n)
    copyto!(temp, mn+1, Iterators.reverse(x), 1, n)
    @inbounds for j in n+1:mn
        temp[j, 1] = zero(T)
        temp[j, 2] = zero(T)
    end
    plan*temp
    temp .*= th_dft
    iplan*temp

    if iszero(β)
        @inbounds @simd for i in 1:m
            y[i] = α * (real(temp[i, 1])+real(temp[i, 2]))
        end
    else
        @inbounds @simd for i in 1:m
            y[i] = α * (real(temp[i, 1])+real(temp[i, 2])) + β*y[i]
        end
    end
    return y
end

# y ← A' x α + y β
function mul!(y::StridedVector{T}, A::Adjoint{T, <:ToeplitzPlusHankel{T}}, x::StridedVector{T}, α::S, β::S) where {T <: Real, S <: Real}
    m, n = size(A)
    @assert m == length(y)
    @assert n == length(x)
    mn = m+n
    AP = A.parent
    tht_dft = AP.tht_dft
    temp = AP.temp
    plan = AP.plan
    iplan = AP.iplan

    copyto!(temp, 1, x, 1, n)
    copyto!(temp, mn+1, Iterators.reverse(x), 1, n)
    @inbounds for j in n+1:mn
        temp[j, 1] = zero(T)
        temp[j, 2] = zero(T)
    end
    plan*temp
    temp .*= tht_dft
    iplan*temp

    if iszero(β)
        @inbounds @simd for i in 1:m
            y[i] = α * (real(temp[i, 1])+real(temp[i, 2]))
        end
    else
        @inbounds @simd for i in 1:m
            y[i] = α * (real(temp[i, 1])+real(temp[i, 2])) + β*y[i]
        end
    end
    return y
end


# C ← A B α + C β
function mul!(C::StridedMatrix{T}, A::ToeplitzPlusHankel{T}, B::StridedMatrix{T}, α::S, β::S) where {T <: Real, S <: Real}
    m, n = size(A)
    @assert m == size(C, 1)
    @assert n == size(B, 1)
    p = size(B, 2)
    if size(C, 2) != p
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end

    th_dft = A.th_dft
    TC = promote_type(float(T), Complex{Float32})
    temp = zeros(TC, m+n, 2p)
    plan = plan_fft!(temp, 1)

    for k in 1:p
        copyto!(view(temp, :, 2k-1), 1, view(B, :, k), 1, n)
        copyto!(view(temp, :, 2k), 1, Iterators.reverse(view(B, :, k)), 1, n)
    end
    plan*temp
    for k in 1:p
        vt = view(temp, :, 2k-1:2k)
        vt .*= th_dft
    end
    plan\temp

    if iszero(β)
        @inbounds for k in 1:p
            for i in 1:m
                C[i, k] = α * (real(temp[i, 2k-1])+real(temp[i, 2k]))
            end
        end
    else
        @inbounds for k in 1:p
            for i in 1:m
                C[i, k] = α * (real(temp[i, 2k-1])+real(temp[i, 2k])) + β*C[i, k]
            end
        end
    end
    return C
end

# Morally equivalent to mul!(C', B', A', α, β)' with StridedMatrix replaced by AbstractMatrix below
function mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::ToeplitzPlusHankel{T}, α::S, β::S) where {T <: Real, S <: Real}
    n, m = size(B)
    @assert m == size(C, 2)
    @assert n == size(A, 2)
    p = size(A, 1)
    if size(C, 1) != p
        throw(DimensionMismatch("input and output matrices must have same number of rows"))
    end

    tht_dft = B.tht_dft
    TC = promote_type(float(T), Complex{Float32})
    temp = zeros(TC, m+n, 2p)
    plan = plan_fft!(temp, 1)

    for k in 1:p
        copyto!(view(temp, :, 2k-1), 1, view(A, k, :), 1, n)
        copyto!(view(temp, :, 2k), 1, Iterators.reverse(view(A, k, :)), 1, n)
    end
    plan*temp
    for k in 1:p
        vt = view(temp, :, 2k-1:2k)
        vt .*= tht_dft
    end
    plan\temp

    if iszero(β)
        @inbounds for k in 1:p
            for i in 1:m
                C[k, i] = α * (real(temp[i, 2k-1])+real(temp[i, 2k]))
            end
        end
    else
        @inbounds for k in 1:p
            for i in 1:m
                C[k, i] = α * (real(temp[i, 2k-1])+real(temp[i, 2k])) + β*C[k, i]
            end
        end
    end
    return C
end

# C ← A' B α + C β
function mul!(C::StridedMatrix{T}, A::Adjoint{T, <:ToeplitzPlusHankel{T}}, B::StridedMatrix{T}, α::S, β::S) where {T <: Real, S <: Real}
    m, n = size(A)
    @assert m == size(C, 1)
    @assert n == size(B, 1)
    p = size(B, 2)
    if size(C, 2) != p
        throw(DimensionMismatch("input and output matrices must have same number of columns"))
    end

    tht_dft = A.parent.tht_dft
    TC = promote_type(float(T), Complex{Float32})
    temp = zeros(TC, m+n, 2p)
    plan = plan_fft!(temp, 1)

    for k in 1:p
        copyto!(view(temp, :, 2k-1), 1, view(B, :, k), 1, n)
        copyto!(view(temp, :, 2k), 1, Iterators.reverse(view(B, :, k)), 1, n)
    end
    plan*temp
    for k in 1:p
        vt = view(temp, :, 2k-1:2k)
        vt .*= tht_dft
    end
    plan\temp

    if iszero(β)
        @inbounds for k in 1:p
            for i in 1:m
                C[i, k] = α * (real(temp[i, 2k-1])+real(temp[i, 2k]))
            end
        end
    else
        @inbounds for k in 1:p
            for i in 1:m
                C[i, k] = α * (real(temp[i, 2k-1])+real(temp[i, 2k])) + β*C[i, k]
            end
        end
    end
    return C
end

# Estimate the Frobenius norm of the Toeplitz-plus-Hankel matrix by working with the symbols.
function normest(A::ToeplitzPlusHankel{T}) where T
    m, n = size(A)
    tc = A.tc
    tr = A.tr
    h = A.h
    ret1 = zero(T)
    ret2 = zero(T)
    if m == min(m, n)
        for i = 1:m
            ret1 += (m+1-i)*abs2(tc[i])
        end
        for i = 2:n-m
            ret1 += m*abs2(tr[i])
        end
        for i = max(n-m+1, 2):n
            ret1 += (n+1-i)*abs2(tr[i])
        end
        for i = 1:m
            ret2 += i*abs2(h[i])
        end
        for i = m+1:n
            ret2 += m*abs2(h[i])
        end
        for i = n+1:m+n-1
            ret2 += (m+n-i)*abs2(h[i])
        end
    else
        for i = 1:n
            ret1 += (n+1-i)*abs2(tr[i])
        end
        for i = 2:m-n
            ret1 += n*abs2(tc[i])
        end
        for i = max(m-n+1, 2):m
            ret1 += (m+1-i)*abs2(tc[i])
        end
        for i = 1:n
            ret2 += i*abs2(h[i])
        end
        for i = n+1:m
            ret2 += n*abs2(h[i])
        end
        for i = m+1:m+n-1
            ret2 += (m+n-i)*abs2(h[i])
        end
    end
    sqrt(ret1) + sqrt(ret2)
end

normest(A::Symmetric{T, <: ToeplitzPlusHankel{T}}) where T = normest(parent(A))+1
normest(A::Hermitian{T, <: ToeplitzPlusHankel{T}}) where T = normest(parent(A))
normest(A::ChebyshevGramMatrix{T}) where T = normest(ToeplitzPlusHankel(A))
