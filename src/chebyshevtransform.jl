## Transforms take values at Chebyshev points of the first and second kinds and produce Chebyshev coefficients

abstract type ChebyshevPlan{T} <: Plan{T} end

*(P::ChebyshevPlan{T}, x::AbstractArray{T}) where T = error("Plan applied to wrong size array")

size(P::ChebyshevPlan) = isdefined(P, :plan) ? size(P.plan) : (0,)
length(P::ChebyshevPlan) = isdefined(P, :plan) ? length(P.plan) : 0


const FIRSTKIND = FFTW.REDFT10
const SECONDKIND = FFTW.REDFT00

struct ChebyshevTransformPlan{T,kind,K,inplace,N,R} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}
    ChebyshevTransformPlan{T,kind,K,inplace,N,R}(plan) where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}(plan)
    ChebyshevTransformPlan{T,kind,K,inplace,N,R}() where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}()
end

ChebyshevTransformPlan{T,kind}(plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}) where {T,kind,K,inplace,N,R} =
    ChebyshevTransformPlan{T,kind,K,inplace,N,R}(plan)

# jump through some hoops to make inferrable

_fftKtype(::Val{N}, _...) where N = Vector{Int32}

function plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        ChebyshevTransformPlan{T,1,_fftKtype(Val{N}(), dims...),true,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        ChebyshevTransformPlan{T,1}(FFTW.plan_r2r!(x, FIRSTKIND, dims...; kws...))
    end
end
function plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    ChebyshevTransformPlan{T,2}(FFTW.plan_r2r!(x, SECONDKIND, dims...; kws...))
end


function plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        ChebyshevTransformPlan{T,1,_fftKtype(Val{N}(), dims...),false,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        ChebyshevTransformPlan{T,1}(FFTW.plan_r2r(x, FIRSTKIND, dims...; kws...))
    end
end
function plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    ChebyshevTransformPlan{T,2}(FFTW.plan_r2r(x, SECONDKIND, dims...; kws...))
end


# convert x if necessary
@inline _plan_mul!(y::AbstractArray{T}, P::Plan{T}, x::StridedArray{T}) where T = mul!(y, P, x)
@inline _plan_mul!(y::AbstractArray{T}, P::Plan{T}, x::AbstractArray) where T = mul!(y, P, convert(Array{T}, x))


for op in (:ldiv, :lmul)
    op_dim_begin! = Symbol(string(op) * "_dim_begin!")
    op_dim_end! = Symbol(string(op) * "_dim_end!")
    op! = Symbol(string(op) * "!")
    @eval begin
        function $op_dim_begin!(α, d::Number, y::AbstractArray{<:Any,N}) where N
            # scale just the d-th dimension by permuting it to the first
            ỹ = PermutedDimsArray(y, _permfirst(d, N))
            $op!(α, view(ỹ, 1, ntuple(_ -> :, Val(N-1))...))
        end

        function $op_dim_end!(α, d::Number, y::AbstractArray{<:Any,N}) where N
            # scale just the d-th dimension by permuting it to the first
            ỹ = PermutedDimsArray(y, _permfirst(d, N))
            $op!(α, view(ỹ, size(ỹ,1), ntuple(_ -> :, Val(N-1))...))
        end
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



function *(P::ChebyshevTransformPlan{T,1,K,true,N}, x::AbstractArray{T,N}) where {T,K,N}
    isempty(x) && return x

    y = P.plan*x # will be  === x if in-place
    _cheb1_rescale!(P.plan.region, y)
end

function mul!(y::AbstractArray{T,N}, P::ChebyshevTransformPlan{T,1,K,false,N}, x::AbstractArray{<:Any,N}) where {T,K,N}
    size(y) == size(x) || throw(DimensionMismatch("output must match dimension"))
    isempty(x) && return y
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

function *(P::ChebyshevTransformPlan{T,2,K,true,N}, x::AbstractArray{T,N}) where {T,K,N}
    n = length(x)
    y = P.plan*x # will be  === x if in-place
    _cheb2_rescale!(P.plan.region, y)
end

function mul!(y::AbstractArray{T,N}, P::ChebyshevTransformPlan{T,2,K,false,N}, x::AbstractArray{<:Any,N}) where {T,K,N}
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

struct IChebyshevTransformPlan{T,kind,K,inplace,N,R} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}
    IChebyshevTransformPlan{T,kind,K,inplace,N,R}(plan) where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}(plan)
    IChebyshevTransformPlan{T,kind,K,inplace,N,R}() where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}()
end

IChebyshevTransformPlan{T,kind}(F::FFTW.r2rFFTWPlan{T,K,inplace,N,R}) where {T,kind,K,inplace,N,R} =
    IChebyshevTransformPlan{T,kind,K,inplace,N,R}(F)



# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
inv(P::ChebyshevTransformPlan{T,2}) where {T} = IChebyshevTransformPlan{T,2}(P.plan)
inv(P::IChebyshevTransformPlan{T,2}) where {T} = ChebyshevTransformPlan{T,2}(P.plan)

inv(P::ChebyshevTransformPlan{T,1}) where {T} = IChebyshevTransformPlan{T,1}(inv(P.plan).p)
inv(P::IChebyshevTransformPlan{T,1}) where {T} = ChebyshevTransformPlan{T,1}(inv(P.plan).p)



\(P::ChebyshevTransformPlan, x::AbstractArray) = inv(P) * x
\(P::IChebyshevTransformPlan, x::AbstractArray) = inv(P) * x


function plan_ichebyshevtransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        IChebyshevTransformPlan{T,1,_fftKtype(Val{N}(), dims...),true,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        IChebyshevTransformPlan{T,1}(FFTW.plan_r2r!(x, IFIRSTKIND, dims...; kws...))
    end
end

function plan_ichebyshevtransform!(x::AbstractArray{T}, ::Val{2}, dims...; kws...) where T<:fftwNumber
    inv(plan_chebyshevtransform!(x, Val(2), dims...; kws...))
end

function plan_ichebyshevtransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        IChebyshevTransformPlan{T,1,_fftKtype(Val{N}(), dims...),false,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        IChebyshevTransformPlan{T,1}(FFTW.plan_r2r(x, IFIRSTKIND, dims...; kws...))
    end
end

function plan_ichebyshevtransform(x::AbstractArray{T}, ::Val{2}, dims...; kws...) where T<:fftwNumber
    inv(plan_chebyshevtransform(x, Val(2), dims...; kws...))
end

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

function *(P::IChebyshevTransformPlan{T,1,K,true,N}, x::AbstractArray{T,N}) where {T<:fftwNumber,K,N}
    n = length(x)
    n == 0 && return x

    _icheb1_prescale!(P.plan.region, x)
    x = ldiv!(2^length(P.plan.region), P.plan*x)
    x
end

function mul!(y::AbstractArray{T,N}, P::IChebyshevTransformPlan{T,1,K,false,N}, x::AbstractArray{T,N}) where {T<:fftwNumber,K,N}
    size(y) == size(x) || throw(DimensionMismatch("output must match dimension"))
    isempty(x) && return y

    _icheb1_prescale!(P.plan.region, x) # TODO: don't mutate x
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

function *(P::IChebyshevTransformPlan{T,2,K,true,N}, x::AbstractArray{T,N}) where {T<:fftwNumber,K,N}
    n = length(x)

    _icheb2_prescale!(P.plan.region, x)
    x = inv(P)*x
    _icheb2_rescale!(P.plan.region, x)
end

function mul!(y::AbstractArray{T,N}, P::IChebyshevTransformPlan{T,2,K,false,N}, x::AbstractArray{<:Any,N}) where {T<:fftwNumber,K,N}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))

    _icheb2_prescale!(P.plan.region, x)
    _plan_mul!(y, inv(P), x)
    _icheb2_postrescale!(P.plan.region, x)
    _icheb2_rescale!(P.plan.region, y)
end

*(P::IChebyshevTransformPlan{T,kind,K,false,N}, x::AbstractArray{T,N}) where {T,kind,K,N} = mul!(similar(x), P, x)
ichebyshevtransform!(x::AbstractArray, dims...; kwds...) = plan_ichebyshevtransform!(x, dims...; kwds...)*x
ichebyshevtransform(x, dims...; kwds...) = plan_ichebyshevtransform(x, dims...; kwds...)*x


#######
# Chebyshev U
#######

const UFIRSTKIND = FFTW.RODFT10
const USECONDKIND = FFTW.RODFT00

struct ChebyshevUTransformPlan{T,kind,K,inplace,N,R} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}
    ChebyshevUTransformPlan{T,kind,K,inplace,N,R}(plan) where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}(plan)
    ChebyshevUTransformPlan{T,kind,K,inplace,N,R}() where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}()
end

ChebyshevUTransformPlan{T,kind}(plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}) where {T,kind,K,inplace,N,R} =
    ChebyshevUTransformPlan{T,kind,K,inplace,N,R}(plan)


function plan_chebyshevutransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        ChebyshevUTransformPlan{T,1,_fftKtype(Val{N}(), dims...),true,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        ChebyshevUTransformPlan{T,1}(FFTW.plan_r2r!(x, UFIRSTKIND, dims...; kws...))
    end
end
function plan_chebyshevutransform!(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    ChebyshevUTransformPlan{T,2}(FFTW.plan_r2r!(x, USECONDKIND, dims...; kws...))
end

function plan_chebyshevutransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        ChebyshevUTransformPlan{T,1,_fftKtype(Val{N}(), dims...),false,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        ChebyshevUTransformPlan{T,1}(FFTW.plan_r2r(x, UFIRSTKIND, dims...; kws...))
    end
end
function plan_chebyshevutransform(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(dims)
        any(≤(1), size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    else
        for d in dims[1]
            size(x,d) ≤ 1 && throw(ArgumentError("Array must contain at least 2 entries"))
        end
    end
    ChebyshevUTransformPlan{T,2}(FFTW.plan_r2r(x, USECONDKIND, dims...; kws...))
end


_permfirst(d, N) = [d; 1:d-1; d+1:N]

@inline function _chebu1_prescale!(d::Number, X::AbstractArray{T,N}) where {T,N}
    X̃ = PermutedDimsArray(X, _permfirst(d, N))
    m = size(X̃,1)
    X̃ .= (sinpi.(one(T)/(2m) .+ ((1:m) .- one(T))/m) ./ m) .* X̃
    X
end

@inline function _chebu1_prescale!(d, y::AbstractArray)
    for k in d
        _chebu1_prescale!(k, y)
    end
    y
end

@inline function _chebu1_postscale!(d::Number, X::AbstractArray{T,N}) where {T,N}
    X̃ = PermutedDimsArray(X, _permfirst(d, N))
    m = size(X̃,1)
    X̃ .= X̃ ./ (sinpi.(one(T)/(2m) .+ ((1:m) .- one(T))/m) ./ m)
    X
end

@inline function _chebu1_postscale!(d, y::AbstractArray)
    for k in d
        _chebu1_postscale!(k, y)
    end
    y
end

function *(P::ChebyshevUTransformPlan{T,1,K,true,N}, x::AbstractArray{T,N}) where {T,K,N}
    length(x) ≤ 1 && return x
    _chebu1_prescale!(P.plan.region, x)
    P.plan * x
end

function mul!(y::AbstractArray{T}, P::ChebyshevUTransformPlan{T,1,K,false}, x::AbstractArray{T}) where {T,K}
    size(y) == size(x) || throw(DimensionMismatch("output must match dimension"))
    isempty(x) && return y
    _chebu1_prescale!(P.plan.region, x) # Todo don't mutate x
    _plan_mul!(y, P.plan, x)
    _chebu1_postscale!(P.plan.region, x)
    for d in P.plan.region
        size(y,d) == 1 && ldiv!(2, y) # fix doubling
    end
    y
end


@inline function _chebu2_prescale!(d::Number, X::AbstractArray{T,N}) where {T,N}
    X̃ = PermutedDimsArray(X, _permfirst(d, N))
    m = size(X̃,1)
    c = one(T)/ (m+1)
    X̃ .= sinpi.((1:m) .* c) .* X̃
    X
end

@inline function _chebu2_prescale!(d, y::AbstractArray)
    for k in d
        _chebu2_prescale!(k, y)
    end
    y
end


@inline function _chebu2_postscale!(d::Number, X::AbstractArray{T,N}) where {T,N}
    X̃ = PermutedDimsArray(X, _permfirst(d, N))
    m = size(X̃,1)
    c = one(T)/ (m+1)
    X̃ .= X̃ ./ sinpi.((1:m) .* c)
    X
end

@inline function _chebu2_postscale!(d, y::AbstractArray)
    for k in d
        _chebu2_postscale!(k, y)
    end
    y
end

function *(P::ChebyshevUTransformPlan{T,2,K,true,N}, x::AbstractArray{T,N}) where {T,K,N}
    sc = one(T)
    for d in P.plan.region
        sc *= one(T)/(size(x,d)+1)
    end
    _chebu2_prescale!(P.plan.region, x)
    lmul!(sc, P.plan * x)
end

function mul!(y::AbstractArray{T}, P::ChebyshevUTransformPlan{T,2,K,false}, x::AbstractArray{T}) where {T,K}
    sc = one(T)
    for d in P.plan.region
        sc *= one(T)/(size(x,d)+1)
    end
    _chebu2_prescale!(P.plan.region, x) # TODO don't mutate x
    _plan_mul!(y, P.plan, x)
    _chebu2_postscale!(P.plan.region, x)
    lmul!(sc, y)
end

*(P::ChebyshevUTransformPlan{T,kind,K,false,N}, x::AbstractArray{T,N}) where {T,kind,K,N} =
    mul!(similar(x), P, x)

chebyshevutransform!(x::AbstractArray{T}, dims...; kws...) where {T<:fftwNumber} =
    plan_chebyshevutransform!(x, dims...; kws...)*x


"""
    chebyshevutransform(x, ::Val{kind}=Val(1))

transforms from values on a Chebyshev grid of the first or second kind to Chebyshev
coefficients of the 2nd kind (Chebyshev U expansion).
"""
chebyshevutransform(x, dims...; kws...) = plan_chebyshevutransform(x, dims...; kws...)*x


## Inverse transforms take ChebyshevU coefficients and produce values at ChebyshevU points of the first and second kinds
const IUFIRSTKIND = FFTW.RODFT01

struct IChebyshevUTransformPlan{T,kind,K,inplace,N,R} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}
    IChebyshevUTransformPlan{T,kind,K,inplace,N,R}(plan) where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}(plan)
    IChebyshevUTransformPlan{T,kind,K,inplace,N,R}() where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}()
end

IChebyshevUTransformPlan{T,kind}(F::FFTW.r2rFFTWPlan{T,K,inplace,N,R}) where {T,kind,K,inplace,N,R} =
    IChebyshevUTransformPlan{T,kind,K,inplace,N,R}(F)

function plan_ichebyshevutransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        IChebyshevUTransformPlan{T,1,_fftKtype(Val{N}(), dims...),true,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        IChebyshevUTransformPlan{T,1}(FFTW.plan_r2r!(x, IUFIRSTKIND, dims...; kws...))
    end
end
function plan_ichebyshevutransform!(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    IChebyshevUTransformPlan{T,2}(FFTW.plan_r2r!(x, USECONDKIND, dims...))
end

function plan_ichebyshevutransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        IChebyshevUTransformPlan{T,1,_fftKtype(Val{N}(), dims...),false,N,isempty(dims) ? NTuple{N,Int} : typeof(dims[1])}()
    else
        IChebyshevUTransformPlan{T,1}(FFTW.plan_r2r(x, IUFIRSTKIND, dims...; kws...))
    end
end
function plan_ichebyshevutransform(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    any(≤(1),size(x)) && throw(ArgumentError("Array must contain at least 2 entries"))
    IChebyshevUTransformPlan{T,2}(FFTW.plan_r2r(x, USECONDKIND, dims...; kws...))
end


# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
inv(P::ChebyshevUTransformPlan{T,2}) where {T} = IChebyshevUTransformPlan{T,2}(P.plan)
inv(P::IChebyshevUTransformPlan{T,2}) where {T} = ChebyshevUTransformPlan{T,2}(P.plan)

inv(P::ChebyshevUTransformPlan{T,1}) where {T} = IChebyshevUTransformPlan{T,1}(inv(P.plan).p)
inv(P::IChebyshevUTransformPlan{T,1}) where {T} = ChebyshevUTransformPlan{T,1}(inv(P.plan).p)

@inline function _ichebu1_postscale!(d::Number, X::AbstractArray{T,N}) where {T,N}
    X̃ = PermutedDimsArray(X, _permfirst(d, N))
    m = size(X̃,1)
    X̃ .= X̃ ./ (2 .* sinpi.(one(T)/(2m) .+ ((1:m) .- one(T))/m))
    X
end


@inline function _ichebu1_postscale!(d, y::AbstractArray)
    for k in d
        _ichebu1_postscale!(k, y)
    end
    y
end

function *(P::IChebyshevUTransformPlan{T,1,K,true}, x::AbstractArray{T}) where {T<:fftwNumber,K}
    length(x) ≤ 1 && return x
    x = P.plan * x
    _ichebu1_postscale!(P.plan.region, x)
end

function mul!(y::AbstractArray{T}, P::IChebyshevUTransformPlan{T,1,K,false}, x::AbstractArray{T}) where {T<:fftwNumber,K}
    size(y) == size(x) || throw(DimensionMismatch("output must match dimension"))
    isempty(x) && return y
    _plan_mul!(y, P.plan, x)
    _ichebu1_postscale!(P.plan.region, y)
    for d in P.plan.region
        size(y,d) == 1 && lmul!(2, y) # fix doubling
    end
    y
end

function _ichebu2_rescale!(d::Number, x::AbstractArray{T}) where T
    _chebu2_postscale!(d, x)
    ldiv!(2, x)
    x
end

@inline function _ichebu2_rescale!(d, y::AbstractArray)
    for k in d
        _ichebu2_rescale!(k, y)
    end
    y
end

function *(P::IChebyshevUTransformPlan{T,2,K,true}, x::AbstractArray{T}) where {T<:fftwNumber,K}
    n = length(x)
    n ≤ 1 && return x

    x = P.plan * x
    _ichebu2_rescale!(P.plan.region, x)
end

function mul!(y::AbstractArray{T}, P::IChebyshevUTransformPlan{T,2,K,false}, x::AbstractArray{T}) where {T<:fftwNumber,K}
    size(y) == size(x) || throw(DimensionMismatch("output must match dimension"))
    length(x) ≤ 1 && return x

    _plan_mul!(y, P.plan, x)
    _ichebu2_rescale!(P.plan.region, y)
end

ichebyshevutransform!(x::AbstractArray{T}, dims...; kwds...) where {T<:fftwNumber} =
    plan_ichebyshevutransform!(x, dims...; kwds...)*x

ichebyshevutransform(x, dims...; kwds...) = plan_ichebyshevutransform(x, dims...; kwds...)*x

*(P::IChebyshevUTransformPlan{T,k,K,false,N}, x::AbstractArray{T,N}) where {T,k,K,N} =
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


###
# BigFloat
# Use `Nothing` and fall back to FFT
###


plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    ChebyshevTransformPlan{T,kind,Nothing,false,N,UnitRange{Int}}()
plan_ichebyshevtransform(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    IChebyshevTransformPlan{T,kind,Nothing,false,N,UnitRange{Int}}()

plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    ChebyshevTransformPlan{T,kind,Nothing,true,N,UnitRange{Int}}()
plan_ichebyshevtransform!(x::AbstractArray{T,N}, ::Val{kind}, dims...; kws...) where {T,N,kind} =
    IChebyshevTransformPlan{T,kind,Nothing,true,N,UnitRange{Int}}()


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


for pln in (:plan_chebyshevtransform!, :plan_chebyshevtransform, 
            :plan_chebyshevutransform!, :plan_chebyshevutransform, 
            :plan_ichebyshevutransform, :plan_ichebyshevutransform!, 
            :plan_ichebyshevtransform, :plan_ichebyshevtransform!)
    @eval begin
        $pln(x::AbstractArray, dims...; kws...) = $pln(x, Val(1), dims...; kws...)
        $pln(::Type{T}, szs, dims...; kwds...) where T = $pln(Array{T}(undef, szs...), dims...; kwds...)
    end
end