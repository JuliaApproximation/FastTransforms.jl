## Transforms take values at Chebyshev points of the first and second kinds and produce Chebyshev coefficients

abstract type ChebyshevPlan{T} <: Plan{T} end

size(P::ChebyshevPlan) = isdefined(P, :plan) ? size(P.plan) : (0,)
length(P::ChebyshevPlan) = isdefined(P, :plan) ? length(P.plan) : 0


const FIRSTKIND = FFTW.REDFT10
const SECONDKIND = FFTW.REDFT00

struct ChebyshevTransformPlan{T,kind,K,inplace,N,P} <: ChebyshevPlan{T}
    plan::P
    function ChebyshevTransformPlan{T,kind,K,inplace,N,P}(plan::P) where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}(plan)
    end
    function ChebyshevTransformPlan{T,kind,K,inplace,N,P}() where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}()
    end
end

ChebyshevTransformPlan{T,kind,K,inplace,N}(plan::P) where {T,kind,K,inplace,N,P} =
    ChebyshevTransformPlan{T,kind,K,inplace,N,P}(plan)

# jump through some hoops to make inferrable

isinplace(::typeof(FFTW.plan_r2r)) = false
isinplace(::typeof(FFTW.plan_r2r!)) = true

function createplan(::Type{CP}, x::AbstractArray{T,N}, ::Val{kind},
            planfn::F, rfftkind, dims...; kws...) where {CP,T<:fftwNumber,N,F,kind}

    inplace = isinplace(planfn)
    if isempty(x)
        flags = get(kws, :flags, FFTW.ESTIMATE)
        szA = size(x) .+ 1
        A = if flags & FFTW.ESTIMATE != 0
            FFTW.FakeArray{T}(szA)
        else
            Array{T}(undef, szA)
        end
        plan = planfn(A, rfftkind, dims...; kws...)
        CP{T,kind,:fftw,inplace,N,typeof(plan)}()
    else
        plan = planfn(x, rfftkind, dims...; kws...)
        CP{T,kind,:fftw,inplace,N}(plan)
    end
end

function plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(ChebyshevTransformPlan, x, Val(1), FFTW.plan_r2r!, FIRSTKIND, dims...; kws...)
end
function plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    createplan(ChebyshevTransformPlan, x, Val(2), FFTW.plan_r2r!, SECONDKIND, dims...; kws...)
end


function plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(ChebyshevTransformPlan, x, Val(1), FFTW.plan_r2r, FIRSTKIND, dims...; kws...)
end
function plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    createplan(ChebyshevTransformPlan, x, Val(2), FFTW.plan_r2r, SECONDKIND, dims...; kws...)
end

plan_chebyshevtransform!(x::AbstractArray, dims...; kws...) = plan_chebyshevtransform!(x, Val(1), dims...; kws...)
plan_chebyshevtransform(x::AbstractArray, dims...; kws...) = plan_chebyshevtransform(x, Val(1), dims...; kws...)


# convert x if necessary
@inline _plan_mul!(y::AbstractArray{T}, P::Plan{T}, x::StridedArray{T}) where T = mul!(y, P, x)
@inline _plan_mul!(y::AbstractArray{T}, P::Plan{T}, x::AbstractArray) where T = mul!(y, P, convert(Array{T}, x))



ldiv_dim_begin!(α, d::Number, y::AbstractVector) = y[1] /= α
function ldiv_dim_begin!(α, d::Number, y::AbstractMatrix)
    if isone(d)
        ldiv!(α, @view(y[1,:]))
    else
        ldiv!(α, @view(y[:,1]))
    end
end
function ldiv_dim_begin!(α, d::Number, y::AbstractArray{<:Any,3})
    if isone(d)
        ldiv!(α, @view(y[1,:,:]))
    elseif d == 2
        ldiv!(α, @view(y[:,1,:]))
    else # d == 3
        ldiv!(α, @view(y[:,:,1]))
    end
end

ldiv_dim_end!(α, d::Number, y::AbstractVector) = y[end] /= α
function ldiv_dim_end!(α, d::Number, y::AbstractMatrix)
    if isone(d)
        ldiv!(α, @view(y[end,:]))
    else
        ldiv!(α, @view(y[:,end]))
    end
end
function ldiv_dim_end!(α, d::Number, y::AbstractArray{<:Any,3})
    if isone(d)
        ldiv!(α, @view(y[end,:,:]))
    elseif d == 2
        ldiv!(α, @view(y[:,end,:]))
    else # d == 3
        ldiv!(α, @view(y[:,:,end]))
    end
end

lmul_dim_begin!(α, d::Number, y::AbstractVector) = y[1] *= α
function lmul_dim_begin!(α, d::Number, y::AbstractMatrix)
    if isone(d)
        lmul!(α, @view(y[1,:]))
    else
        lmul!(α, @view(y[:,1]))
    end
end
function lmul_dim_begin!(α, d::Number, y::AbstractArray{<:Any,3})
    if isone(d)
        lmul!(α, @view(y[1,:,:]))
    elseif d == 2
        lmul!(α, @view(y[:,1,:]))
    else # d == 3
        lmul!(α, @view(y[:,:,1]))
    end
end

lmul_dim_end!(α, d::Number, y::AbstractVector) = y[end] *= α
function lmul_dim_end!(α, d::Number, y::AbstractMatrix)
    if isone(d)
        lmul!(α, @view(y[end,:]))
    else
        lmul!(α, @view(y[:,end]))
    end
end
function lmul_dim_end!(α, d::Number, y::AbstractArray{<:Any,3})
    if isone(d)
        lmul!(α, @view(y[end,:,:]))
    elseif d == 2
        lmul!(α, @view(y[:,end,:]))
    else # d == 3
        lmul!(α, @view(y[:,:,end]))
    end
end


@inline function _cheb1_rescale!(d::Number, y::AbstractArray)
    ldiv_dim_begin!(2, d, y)
    ldiv!(size(y,d), y)
end

function _prod_size(sz, d)
    ret = 1
    for k in d
        ret *= sz[k]
    end
    ret
end


@inline function _cheb1_rescale!(d, y::AbstractArray)
    for k in d
        ldiv_dim_begin!(2, k, y)
    end
    ldiv!(_prod_size(size(y), d), y)
end

function *(P::ChebyshevTransformPlan{T,1,:fftw,true,N}, x::AbstractArray{T,N}) where {T,N}
    n = length(x)
    n == 0 && return x

    y = P.plan*x # will be  === x if in-place
    _cheb1_rescale!(P.plan.region, y)
end

function mul!(y::AbstractArray{T,N}, P::ChebyshevTransformPlan{T,1,:fftw,false,N}, x::AbstractArray{<:Any,N}) where {T,N}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))
    n == 0 && return y
    _plan_mul!(y, P.plan, x)
    _cheb1_rescale!(P.plan.region, y)
end



function _cheb2_rescale!(d::Number, y::AbstractArray)
    ldiv_dim_begin!(2, d, y)
    ldiv_dim_end!(2, d, y)
    ldiv!(size(y,d)-1, y)
end

# TODO: higher dimensional arrays
function _cheb2_rescale!(d, y::AbstractArray)
    for k in d
        ldiv_dim_begin!(2, k, y)
        ldiv_dim_end!(2, k, y)
    end

    ldiv!(_prod_size(size(y) .- 1, d), y)
end

function *(P::ChebyshevTransformPlan{T,2,:fftw,true,N}, x::AbstractArray{T,N}) where {T<:fftwNumber,N}
    n = length(x)
    y = P.plan*x # will be  === x if in-place
    _cheb2_rescale!(P.plan.region, y)
end

function mul!(y::AbstractArray{T,N}, P::ChebyshevTransformPlan{T,2,:fftw,false,N}, x::AbstractArray{<:Any,N}) where {T<:fftwNumber,N}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))
    _plan_mul!(y, P.plan, x)
    _cheb2_rescale!(P.plan.region, y)
end

*(P::ChebyshevTransformPlan{T,kind,K,false,N}, x::AbstractArray{T,N}) where {T,kind,K,N} =
    mul!(similar(x), P, x)

"""
    chebyshevtransform!(x, kind=Val(1))

transforms from values on a Chebyshev grid of the first or second kind to Chebyshev
coefficients, in-place
"""
chebyshevtransform!(x, dims...; kws...) = plan_chebyshevtransform!(x, dims...; kws...)*x


"""
    chebyshevtransform(x, kind=Val(1))

transforms from values on a Chebyshev grid of the first or second kind to Chebyshev
coefficients.
"""
chebyshevtransform(x, dims...; kws...) = plan_chebyshevtransform(x, dims...; kws...) * x


## Inverse transforms take Chebyshev coefficients and produce values at Chebyshev points of the first and second kinds


const IFIRSTKIND = FFTW.REDFT01

struct IChebyshevTransformPlan{T,kind,K,inplace,N,P} <: ChebyshevPlan{T}
    plan::P
    function IChebyshevTransformPlan{T,kind,K,inplace,N,P}(plan::P) where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}(plan)
    end
    function IChebyshevTransformPlan{T,kind,K,inplace,N,P}() where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}()
    end
end

IChebyshevTransformPlan{T,kind,K,inplace,N}(F::P) where {T,kind,K,inplace,N,P} =
    IChebyshevTransformPlan{T,kind,K,inplace,N,P}(F)


# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
function inv(P::ChebyshevTransformPlan{T,2,K,inplace,N}) where {T,K,inplace,N}
    IChebyshevTransformPlan{T,2,K,inplace,N}(P.plan)
end
function inv(P::IChebyshevTransformPlan{T,2,K,inplace,N}) where {T,K,inplace,N}
    ChebyshevTransformPlan{T,2,K,inplace,N}(P.plan)
end

function inv(P::ChebyshevTransformPlan{T,1,K,inplace,N}) where {T,K,inplace,N}
    IChebyshevTransformPlan{T,1,K,inplace,N}(inv(P.plan).p)
end
function inv(P::IChebyshevTransformPlan{T,1,K,inplace,N}) where {T,K,inplace,N}
    ChebyshevTransformPlan{T,1,K,inplace,N}(inv(P.plan).p)
end



\(P::ChebyshevTransformPlan, x::AbstractArray) = inv(P) * x
\(P::IChebyshevTransformPlan, x::AbstractArray) = inv(P) * x


function plan_ichebyshevtransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(IChebyshevTransformPlan, x, Val(1), FFTW.plan_r2r!, IFIRSTKIND, dims...; kws...)
end

function plan_ichebyshevtransform!(x::AbstractArray{T}, ::Val{2}, dims...; kws...) where T<:fftwNumber
    inv(plan_chebyshevtransform!(x, Val(2), dims...; kws...))
end

function plan_ichebyshevtransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(IChebyshevTransformPlan, x, Val(1), FFTW.plan_r2r, IFIRSTKIND, dims...; kws...)
end

function plan_ichebyshevtransform(x::AbstractArray{T}, ::Val{2}, dims...; kws...) where T<:fftwNumber
    inv(plan_chebyshevtransform(x, Val(2), dims...; kws...))
end

plan_ichebyshevtransform!(x::AbstractArray, dims...; kws...) = plan_ichebyshevtransform!(x, Val(1), dims...; kws...)
plan_ichebyshevtransform(x::AbstractArray, dims...; kws...) = plan_ichebyshevtransform(x, Val(1), dims...; kws...)

@inline function _icheb1_prescale!(d::Number, x::AbstractArray)
    lmul_dim_begin!(2, d, x)
    x
end
@inline function _icheb1_prescale!(d, x::AbstractArray)
    for k in d
        _icheb1_prescale!(k, x)
    end
    x
end
@inline function _icheb1_postscale!(d::Number, x::AbstractArray)
    ldiv_dim_begin!(2, d, x)
    x
end

@inline function _icheb1_postscale!(d, x::AbstractArray)
    for k in d
        _icheb1_postscale!(k, x)
    end
    x
end

function *(P::IChebyshevTransformPlan{T,1,:fftw,true,N}, x::AbstractArray{T,N}) where {T,N}
    n = length(x)
    n == 0 && return x

    _icheb1_prescale!(P.plan.region, x)
    x = ldiv!(2^length(P.plan.region), P.plan*x)
    x
end

function mul!(y::AbstractArray{T,N}, P::IChebyshevTransformPlan{T,1,:fftw,false,N}, x::AbstractArray{T,N}) where {T<:fftwNumber,N}
    size(y) == size(x) || throw(DimensionMismatch("output must match dimension"))
    isempty(x) && return y

    _icheb1_prescale!(P.plan.region, x) # Todo: don't mutate x
    _plan_mul!(y, P.plan, x)
    _icheb1_postscale!(P.plan.region, x)
    ldiv!(2^length(P.plan.region), y)
end

@inline function _icheb2_prescale!(d::Number, x::AbstractArray)
    lmul_dim_begin!(2, d, x)
    lmul_dim_end!(2, d, x)
    x
end
@inline function _icheb2_prescale!(d, x::AbstractArray)
    for k in d
        _icheb2_prescale!(k, x)
    end
    x
end

@inline function _icheb2_postrescale!(d::Number, x::AbstractArray)
    ldiv_dim_begin!(2, d, x)
    ldiv_dim_end!(2, d, x)
    x
end
@inline function _icheb2_postrescale!(d, x::AbstractArray)
    for k in d
        _icheb2_postrescale!(k, x)
    end
    x
end
@inline function _icheb2_rescale!(d::Number, y::AbstractArray{T}) where T
    _icheb2_prescale!(d, y)
    lmul!(convert(T, size(y,d) - 1)/2, y)
    y
end
@inline function _icheb2_rescale!(d, y::AbstractArray{T}) where T
    _icheb2_prescale!(d, y)
    lmul!(_prod_size(convert.(T, size(y) .- 1)./2, d), y)
    y
end

function *(P::IChebyshevTransformPlan{T,2,:fftw,true,N}, x::AbstractArray{T,N}) where {T<:fftwNumber,N}
    n = length(x)

    _icheb2_prescale!(P.plan.region, x)
    x = inv(P)*x
    _icheb2_rescale!(P.plan.region, x)
end

function mul!(y::AbstractArray{T,N}, P::IChebyshevTransformPlan{T,2,:fftw,false,N}, x::AbstractArray{<:Any,N}) where {T<:fftwNumber,N}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))

    _icheb2_prescale!(P.plan.region, x)
    _plan_mul!(y, inv(P), x)
    _icheb2_postrescale!(P.plan.region, x)
    _icheb2_rescale!(P.plan.region, y)
end

*(P::IChebyshevTransformPlan{T,kind,:fftw,false,N}, x::AbstractArray{T,N}) where {T<:fftwNumber,kind,N} = mul!(similar(x), P, x)
ichebyshevtransform!(x::AbstractArray, dims...; kwds...) = plan_ichebyshevtransform!(x, dims...; kwds...)*x
ichebyshevtransform(x, dims...; kwds...) = plan_ichebyshevtransform(x, dims...; kwds...)*x


## Chebyshev U

const UFIRSTKIND = FFTW.RODFT10
const USECONDKIND = FFTW.RODFT00

struct ChebyshevUTransformPlan{T,kind,K,inplace,N,P} <: ChebyshevPlan{T}
    plan::P
    function ChebyshevUTransformPlan{T,kind,K,inplace,N,P}(plan::P) where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}(plan)
    end
    function ChebyshevUTransformPlan{T,kind,K,inplace,N,P}() where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}()
    end
end

ChebyshevUTransformPlan{T,kind,K,inplace,N}(plan::P) where {T,kind,K,inplace,N,P} =
    ChebyshevUTransformPlan{T,kind,K,inplace,N,P}(plan)


function plan_chebyshevutransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(ChebyshevUTransformPlan, x, Val(1), FFTW.plan_r2r!, UFIRSTKIND, dims...; kws...)
end
function plan_chebyshevutransform!(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    createplan(ChebyshevUTransformPlan, x, Val(2), FFTW.plan_r2r!, USECONDKIND, dims...; kws...)
end

function plan_chebyshevutransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(ChebyshevUTransformPlan, x, Val(1), FFTW.plan_r2r, UFIRSTKIND, dims...; kws...)
end
function plan_chebyshevutransform(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    createplan(ChebyshevUTransformPlan, x, Val(2), FFTW.plan_r2r, USECONDKIND, dims...; kws...)
end

plan_chebyshevutransform!(x::AbstractArray, dims...; kws...) = plan_chebyshevutransform!(x, Val(1), dims...; kws...)
plan_chebyshevutransform(x::AbstractArray, dims...; kws...) = plan_chebyshevutransform(x, Val(1), dims...; kws...)


@inline function _chebu1_prescale!(_, x::AbstractVector{T}) where T
    n = length(x)
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] *= sinpi(one(T)/(2n) + (k-one(T))/n)/n
    end
    x
end

@inline function _chebu1_postscale!(_, x::AbstractVector{T}) where T
    n = length(x)
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= sinpi(one(T)/(2n) + (k-one(T))/n)/n
    end
    x
end

function *(P::ChebyshevUTransformPlan{T,1,:fftw,true}, x::AbstractVector{T}) where {T}
    length(x) ≤ 1 && return x
    _chebu1_prescale!(P.plan.region, x)
    P.plan * x
end

function mul!(y::AbstractVector{T}, P::ChebyshevUTransformPlan{T,1,:fftw,false}, x::AbstractVector{T}) where {T}
    n = length(x)
    length(x) ≤ 1 && return copyto!(y, x)
    _chebu1_prescale!(P.plan.region, x)
    _plan_mul!(y, P.plan, x)
    _chebu1_postscale!(P.plan.region, x)
    y
end

@inline function _chebu2_prescale!(_, x::AbstractVector{T}) where T
    n = length(x)
    c = one(T)/ (n+1)
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] *= sinpi(k*c)
    end
    x
end

@inline function _chebu2_postscale!(_, x::AbstractVector{T}) where T
    n = length(x)
    c = one(T)/ (n+1)
    @inbounds for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= sinpi(k*c)
    end
    x
end

function *(P::ChebyshevUTransformPlan{T,2,:fftw,true}, x::AbstractVector{T}) where {T}
    n = length(x)
    n ≤ 1 && return x
    _chebu2_prescale!(P.plan.region, x)
    lmul!(one(T)/ (n+1), P.plan * x)
end

function mul!(y::AbstractVector{T}, P::ChebyshevUTransformPlan{T,2,:fftw,false}, x::AbstractVector{T}) where {T}
    n = length(x)
    n ≤ 1 && return copyto!(y, x)
    _chebu2_prescale!(P.plan.region, x)
    _plan_mul!(y, P.plan, x)
    _chebu2_postscale!(P.plan.region, x)
    lmul!(one(T)/ (n+1), y)
end

*(P::ChebyshevUTransformPlan{T,kind,:fftw,false,N}, x::AbstractArray{T,N}) where {T,kind,N} =
    mul!(similar(x), P, x)

chebyshevutransform!(x::AbstractVector{T}, dims...; kws...) where {T<:fftwNumber} =
    plan_chebyshevutransform!(x, dims...; kws...)*x


"""
    chebyshevutransform(x, ::Val{kind}=Val(1))

transforms from values on a Chebyshev grid of the first or second kind to Chebyshev
coefficients of the 2nd kind (Chebyshev U expansion).
"""
chebyshevutransform(x, dims...; kws...) = plan_chebyshevutransform(x, dims...; kws...)*x


## Inverse transforms take ChebyshevU coefficients and produce values at ChebyshevU points of the first and second kinds
const IUFIRSTKIND = FFTW.RODFT01

struct IChebyshevUTransformPlan{T,kind,K,inplace,N,P} <: ChebyshevPlan{T}
    plan::P
    function IChebyshevUTransformPlan{T,kind,K,inplace,N,P}(plan::P) where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}(plan)
    end
    function IChebyshevUTransformPlan{T,kind,K,inplace,N,P}() where {T,kind,K,inplace,N,P}
        new{T,kind,K,inplace,N,P}()
    end
end

IChebyshevUTransformPlan{T,kind,K,inplace,N}(F::P) where {T,kind,K,inplace,N,P} =
    IChebyshevUTransformPlan{T,kind,K,inplace,N,P}(F)

function plan_ichebyshevutransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(IChebyshevUTransformPlan, x, Val(1), FFTW.plan_r2r!, IUFIRSTKIND, dims...; kws...)
end
function plan_ichebyshevutransform!(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    createplan(IChebyshevUTransformPlan, x, Val(2), FFTW.plan_r2r!, USECONDKIND, dims...; kws...)
end

function plan_ichebyshevutransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    createplan(IChebyshevUTransformPlan, x, Val(1), FFTW.plan_r2r, IUFIRSTKIND, dims...; kws...)
end
function plan_ichebyshevutransform(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    createplan(IChebyshevUTransformPlan, x, Val(2), FFTW.plan_r2r, USECONDKIND, dims...; kws...)
end


plan_ichebyshevutransform!(x::AbstractArray, dims...; kws...) = plan_ichebyshevutransform!(x, Val(1), dims...; kws...)
plan_ichebyshevutransform(x::AbstractArray, dims...; kws...) = plan_ichebyshevutransform(x, Val(1), dims...; kws...)

# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
function inv(P::ChebyshevUTransformPlan{T,2,K,inplace,N}) where {T,K,inplace,N}
    IChebyshevUTransformPlan{T,2,K,inplace,N}(P.plan)
end
function inv(P::IChebyshevUTransformPlan{T,2,K,inplace,N}) where {T,K,inplace,N}
    ChebyshevUTransformPlan{T,2,K,inplace,N}(P.plan)
end

function inv(P::ChebyshevUTransformPlan{T,1,K,inplace,N}) where {T,K,inplace,N}
    IChebyshevUTransformPlan{T,1,K,inplace,N}(inv(P.plan).p)
end
function inv(P::IChebyshevUTransformPlan{T,1,K,inplace,N}) where {T,K,inplace,N}
    ChebyshevUTransformPlan{T,1,K,inplace,N}(inv(P.plan).p)
end


function _ichebyu1_postscale!(_, x::AbstractVector{T}) where T
    n = length(x)
    @inbounds for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= 2sinpi(one(T)/(2n) + (k-one(T))/n)
    end
    x
end
function *(P::IChebyshevUTransformPlan{T,1,:fftw,true}, x::AbstractVector{T}) where {T<:fftwNumber}
    n = length(x)
    n ≤ 1 && return x

    x = P.plan * x
    _ichebyu1_postscale!(P.plan.region, x)
end

function mul!(y::AbstractVector{T}, P::IChebyshevUTransformPlan{T,1,:fftw,false}, x::AbstractVector{T}) where {T<:fftwNumber}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))
    n ≤ 1 && return x

    _plan_mul!(y, P.plan, x)
    _ichebyu1_postscale!(P.plan.region, y)
end

function _ichebu2_rescale!(_, x::AbstractVector{T}) where T
    n = length(x)
    c = one(T)/ (n+1)
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= sinpi(k*c)
    end
    ldiv!(2, x)
    x
end

function *(P::IChebyshevUTransformPlan{T,2,:fftw,true}, x::AbstractVector{T}) where {T<:fftwNumber}
    n = length(x)
    n ≤ 1 && return x

    x = P.plan * x
    _ichebu2_rescale!(P.plan.region, x)
end

function mul!(y::AbstractVector{T}, P::IChebyshevUTransformPlan{T,2,:fftw,false}, x::AbstractVector{T}) where {T<:fftwNumber}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))
    n ≤ 1 && return x

    _plan_mul!(y, P.plan, x)
    _ichebu2_rescale!(P.plan.region, y)
end

ichebyshevutransform!(x::AbstractVector{T}, dims...; kwds...) where {T<:fftwNumber} =
    plan_ichebyshevutransform!(x, dims...; kwds...)*x

ichebyshevutransform(x, dims...; kwds...) = plan_ichebyshevutransform(x, dims...; kwds...)*x

*(P::IChebyshevUTransformPlan{T,k,:fftw,false,N}, x::AbstractArray{T,N}) where {T,k,N} =
    mul!(similar(x), P, x)


## Code generation for integer inputs

for func in (:chebyshevtransform,:ichebyshevtransform,:chebyshevutransform,:ichebyshevutransform)
    @eval $func(x::AbstractVector{T}, dims...; kwds...) where {T<:Integer} = $func(convert(AbstractVector{Float64},x), dims...; kwds...)
end



## points

struct ChebyshevGrid{kind,T} <: AbstractVector{T}
    n::Int
    function ChebyshevGrid{1,T}(n::Int) where T
        n ≥ 0 || throw(ArgumentError("Number of points must be nonnehative"))
        new{1,T}(n)
    end
    function ChebyshevGrid{2,T}(n::Int) where T
        n ≥ 2 || throw(ArgumentError("Number of points must be greater than 2"))
        new{2,T}(n)
    end
end

ChebyshevGrid{kind}(n::Integer) where kind = ChebyshevGrid{kind,Float64}(n)

size(g::ChebyshevGrid) = (g.n,)
getindex(g::ChebyshevGrid{1,T}, k::Integer) where T =
    sinpi(convert(T,g.n-2k+1)/(2g.n))

getindex(g::ChebyshevGrid{2,T}, k::Integer) where T =
    sinpi(convert(T,g.n-2k+1)/(2g.n-2))

chebyshevpoints(::Type{T}, n::Integer, ::Val{kind}) where {T<:Number,kind} = ChebyshevGrid{kind,T}(n)
chebyshevpoints(::Type{T}, n::Integer) where T = chebyshevpoints(T, n, Val(1))
chebyshevpoints(n::Integer, kind=Val(1)) = chebyshevpoints(Float64, n, kind)


# sin(nθ) coefficients to values at Clenshaw-Curtis nodes except ±1
#
# struct DSTPlan{T,kind,inplace,P} <: Plan{T}
#     plan::P
# end
#
# DSTPlan{k,inp}(plan) where {k,inp} =
#     DSTPlan{eltype(plan),k,inp,typeof(plan)}(plan)
#
#
# plan_DSTI!(x) = length(x) > 0 ? DSTPlan{1,true}(FFTW.FFTW.plan_r2r!(x, FFTW.FFTW.RODFT00)) :
#                                 fill(one(T),1,length(x))
#
# function *(P::DSTPlan{T,1}, x::AbstractArray) where {T}
#     x = P.plan*x
#     rmul!(x,half(T))
# end


plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    ChebyshevTransformPlan{T,kind,Nothing,false,N,Nothing}()
plan_ichebyshevtransform(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    IChebyshevTransformPlan{T,kind,Nothing,false,N,Nothing}()

plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    ChebyshevTransformPlan{T,kind,Nothing,true,N,Nothing}()
plan_ichebyshevtransform!(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    IChebyshevTransformPlan{T,kind,Nothing,true,N,Nothing}()


#following Chebfun's @Chebtech1/vals2coeffs.m and @Chebtech2/vals2coeffs.m
function *(P::ChebyshevTransformPlan{T,1,Nothing,false}, x::AbstractVector{T}) where T
    n = length(x)
    if n == 1
        x
    else
        w = [2exp(im*convert(T,π)*k/2n) for k=0:n-1]
        ret = w.*ifft([x;reverse(x)])[1:n]
        ret = T<:Real ? real(ret) : ret
        ret[1] /= 2
        ret
    end
end


# function *(P::ChebyshevTransformPlan{T,1,K,Nothing,false}, x::AbstractVector{T}) where {T,K}
#     n = length(x)
#     if n == 1
#         x
#     else
#         ret = ifft([x;x[end:-1:2]])[1:n]
#         ret = T<:Real ? real(ret) : ret
#         ret[2:n-1] *= 2
#         ret
#     end
# end


*(P::ChebyshevTransformPlan{T,1,Nothing,true,N,R}, x::AbstractVector{T}) where {T,N,R} =
    copyto!(x, ChebyshevTransformPlan{T,1,Nothing,false,N,R}() * x)
# *(P::ChebyshevTransformPlan{T,2,true,Nothing}, x::AbstractVector{T}) where T =
#     copyto!(x, ChebyshevTransformPlan{T,2,false,Nothing}() * x)


#following Chebfun's @Chebtech1/vals2coeffs.m and @Chebtech2/vals2coeffs.m
function *(P::IChebyshevTransformPlan{T,1,Nothing,false}, x::AbstractVector{T}) where T
    n = length(x)
    if n == 1
        x
    else
        w = [exp(-im*convert(T,π)*k/2n)/2 for k=0:2n-1]
        w[1] *= 2;w[n+1] *= 0;w[n+2:end] *= -1
        ret = fft(w.*[x;one(T);x[end:-1:2]])
        ret = T<:Real ? real(ret) : ret
        ret[1:n]
    end
end

# function *(P::IChebyshevTransformPlan{T,2,K,Nothing,true}, x::AbstractVector{T}) where {T,K}
#     n = length(x)
#     if n == 1
#         x
#     else
#         x[1] *= 2; x[end] *= 2
#         chebyshevtransform!(x, Val(2))
#         x[1] *= 2; x[end] *= 2
#         lmul!(convert(T,n-1)/2, x)
#         x
#     end
# end

*(P::IChebyshevTransformPlan{T,1,Nothing,true,N,R}, x::AbstractVector{T}) where {T,N,R} =
    copyto!(x, IChebyshevTransformPlan{T,1,Nothing,false,N,R}() * x)
# *(P::IChebyshevTransformPlan{T,SECONDKIND,false,Nothing}, x::AbstractVector{T}) where T =
#     IChebyshevTransformPlan{T,SECONDKIND,true,Nothing}() * copy(x)
