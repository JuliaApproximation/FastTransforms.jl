"""
   forwardrecurrence!(v, A, B, C, x)

evaluates the orthogonal polynomials at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF,
overwriting `v` with the results.   
"""
function forwardrecurrence!(v::AbstractVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, x) where T
    N = length(v)
    N == 0 && return v
    length(A)+1 ≥ N && length(B)+1 ≥ N && length(C)+1 ≥ N || throw(ArgumentError("A, B, C must contain at least $(N-1) entries"))
    p0 = one(T) # assume OPs are normalized to one for no
    p1 = convert(T, N == 1 ? p0 : A[1]x + B[1]) # avoid accessing A[1]/B[1] if empty
    _forwardrecurrence!(v, A, B, C, x, p0, p1)
end


Base.@propagate_inbounds _forwardrecurrence_next(n, A, B, C, x, p0, p1) = muladd(muladd(A[n],x,B[n]), p1, -C[n]*p0)
# special case for B[n] == 0
Base.@propagate_inbounds _forwardrecurrence_next(n, A, ::Zeros, C, x, p0, p1) = muladd(A[n]*x, p1, -C[n]*p0)
# special case for Chebyshev U
Base.@propagate_inbounds _forwardrecurrence_next(n, A::AbstractFill, ::Zeros, C::Ones, x, p0, p1) = muladd(getindex_value(A)*x, p1, -p0)


# this supports adaptivity: we can populate `v` for large `n`
function _forwardrecurrence!(v::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, p0, p1)
    N = length(v)
    N == 0 && return v
    v[1] = p0
    N == 1 && return v
    v[2] = p1
    @inbounds for n = 2:N-1
        p1,p0 = _forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
        v[n+1] = p1
    end
    v
end



forwardrecurrence(N::Integer, A::AbstractVector, B::AbstractVector, C::AbstractVector, x) =
    forwardrecurrence!(Vector{promote_type(eltype(A),eltype(B),eltype(C),typeof(x))}(undef, N), A, B, C, x)


"""
clenshaw!(c, A, B, C, x)

evaluates the orthogonal polynomial expansion with coefficients `c` at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF,
overwriting `x` with the results.
"""
clenshaw!(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector) = 
    clenshaw!(c, A, B, C, x, Ones{eltype(x)}(length(x)), x)


"""
clenshaw!(c, A, B, C, x, ϕ₀, f)

evaluates the orthogonal polynomial expansion with coefficients `c` at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF and ϕ₀ is the zeroth coefficient,
overwriting `f` with the results.
"""
function clenshaw!(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector, ϕ₀::AbstractVector, f::AbstractVector)
    f .= ϕ₀ .* clenshaw.(Ref(c), Ref(A), Ref(B), Ref(C), x)
end


Base.@propagate_inbounds _clenshaw_next(n, A, B, C, x, c, bn1, bn2) = muladd(muladd(A[n],x,B[n]), bn1, muladd(-C[n+1],bn2,c[n]))
Base.@propagate_inbounds _clenshaw_next(n, A, ::Zeros, C, x, c, bn1, bn2) = muladd(A[n]*x, bn1, muladd(-C[n+1],bn2,c[n]))
# Chebyshev U
Base.@propagate_inbounds _clenshaw_next(n, A::AbstractFill, ::Zeros, C::Ones, x, c, bn1, bn2) = muladd(getindex_value(A)*x, bn1, -bn2+c[n])

# allow special casing first arg, for ChebyshevT in OrthogonalPolynomialsQuasi
Base.@propagate_inbounds _clenshaw_first(A, B, C, x, c, bn1, bn2) = muladd(muladd(A[1],x,B[1]), bn1, muladd(-C[2],bn2,c[1]))


"""
    clenshaw(c, A, B, C, x)

evaluates the orthogonal polynomial expansion with coefficients `c` at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF.
`x` may also be a single `Number`.
"""
     
function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),typeof(x))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    N == 0 && return zero(T)
    @inbounds begin
        bn2 = zero(T)
        bn1 = convert(T,c[N])
        for n = N-1:-1:2
            bn1,bn2 = _clenshaw_next(n, A, B, C, x, c, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first(A, B, C, x, c, bn1, bn2)
    end
    bn1
end


clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector) = 
    clenshaw!(c, A, B, C, copy(x))

###
# Chebyshev T special cases
###

"""
   clenshaw!(c, x)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at points `x`,
overwriting `x` with the results.
"""
clenshaw!(c::AbstractVector, x::AbstractVector) = clenshaw!(c, x, x)


"""
   clenshaw!(c, x, f)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at points `x`,
overwriting `f` with the results.
"""
clenshaw!(c::AbstractVector, x::AbstractVector, f::AbstractVector) = _clenshaw!(MemoryLayout(c), MemoryLayout(x), MemoryLayout(f), c, x, f)

function _clenshaw!(_, _, _, c::AbstractVector, x::AbstractVector, f::AbstractVector)
    f .= clenshaw.(Ref(c), x)
end

"""
    clenshaw(c, x)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at  the points `x`.
`x` may also be a single `Number`.
"""
function clenshaw(c::AbstractVector, x::Number)
    N,T = length(c),promote_type(eltype(c),typeof(x))
    if N == 0
        return zero(T)
    elseif N == 1 # avoid issues with NaN x
        return first(c)*one(x)
    end

    y = 2x
    bk1,bk2 = zero(T),zero(T)
    @inbounds begin
        for k = N:-1:2
            bk1,bk2 = muladd(y,bk1,c[k]-bk2),bk1
        end
        muladd(x,bk1,c[1]-bk2)
    end
end

clenshaw(c::AbstractVector, x::AbstractVector) = clenshaw!(c, copy(x))

