"""
   forwardrecurrence!(v, A, B, C, x, shift=0)

evaluates the orthogonal polynomials at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF,
overwriting `v` with the results.   
"""
function forwardrecurrence!(v::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x)
    N = length(v)
    N == 0 && return v
    length(A)+1 ≥ N && length(B)+1 ≥ N && length(C)+1 ≥ N || throw(ArgumentError("A, B, C must contain at least $(N-1) entries"))

    p0 = one(x) # assume OPs are normalized to one for now
    v[1] = p0
    N == 1 && return v
    p1 = A[1]x + B[1]
    v[2] = p1
    @inbounds for n = 2:N-1
        p1,p0 = muladd(muladd(A[n],x,B[n]), p1, -C[n]*p0),p1
        v[n+1] = p1
    end
    v
end


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
    if length(A) < N || length(B) < N || length(C) < N
        throw(ArgumentError("A, B, C must contain at least $N entries"))
    end
    N == 0 && return zero(T)
    @inbounds begin
        bk2 = zero(T)
        bk1 = convert(T,c[N])
        for k = N-1:-1:1
            bk1,bk2 = muladd(muladd(A[k],x,B[k]),bk1,muladd(-C[k],bk2,c[k])),bk1
        end
    end
    bk1
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
function clenshaw!(c::AbstractVector, x::AbstractVector, f::AbstractVector)
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

