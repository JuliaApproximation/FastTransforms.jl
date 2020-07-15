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
    @inbounds for k in axes(x,1)
        f[k] = clenshaw(c, x[k])
    end
    f
end

"""
    clenshaw(c, x)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at  the points `x`.
`x` may also be a single `Number`.
"""
function clenshaw(c::AbstractVector, x::Number)
    N,T = length(c),promote_type(eltype(c),typeof(x))
    if N == 0
        return zero(x)
    elseif N == 1 # avoid issues with NaN x
        return first(c)*one(x)
    end

    y = 2x
    bk1,bk2 = zero(T),zero(T)
    @inbounds for k = N:-1:2
        bk2, bk1 = bk1, muladd(y,bk1,c[k]-bk2)
    end

    muladd(x,bk1,c[1]-bk2)
end


clenshaw(c::AbstractVector, x::AbstractVector) = clenshaw!(c, copy(x))

