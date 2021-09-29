## Transforms take values at Chebyshev points of the first and second kinds and produce Chebyshev coefficients

abstract type ChebyshevPlan{T} <: Plan{T} end

size(P::ChebyshevPlan) = isdefined(P, :plan) ? size(P.plan) : (0,)
length(P::ChebyshevPlan) = isdefined(P, :plan) ? length(P.plan) : 0


const FIRSTKIND = 5
const SECONDKIND = 3

struct ChebyshevTransformPlan{T,kind,K,inplace,N,R} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}
    ChebyshevTransformPlan{T,kind,K,inplace,N,R}(plan) where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}(plan)
    ChebyshevTransformPlan{T,kind,K,inplace,N,R}() where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}()
end

ChebyshevTransformPlan{T,kind,K}(plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}) where {T,kind,K,inplace,N,R} =
    ChebyshevTransformPlan{T,kind,K,inplace,N,R}(plan)

# jump through some hoops to make inferrable
@inline kindtuple(KIND,N) = ntuple(_ -> KIND,N)
@inline kindtuple(KIND,N,::Integer) = (KIND,)
function plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        ChebyshevTransformPlan{T,1,kindtuple(FIRSTKIND,N,dims...),true,N,isempty(dims) ? UnitRange{Int} : typeof(dims)}()
    else
        ChebyshevTransformPlan{T,1,kindtuple(FIRSTKIND,N,dims...)}(FFTW.plan_r2r!(x, FFTW.REDFT10, dims...; kws...))
    end
end
function plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    length(x) ≤ 1 && throw(ArgumentError("Array must contain at least 2 entries"))
    ChebyshevTransformPlan{T,2,kindtuple(SECONDKIND,N,dims...)}(FFTW.plan_r2r!(x, FFTW.REDFT00, dims...; kws...))
end


function plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        ChebyshevTransformPlan{T,1,kindtuple(FIRSTKIND,N,dims...),false,N,isempty(dims) ? UnitRange{Int} : typeof(dims)}()
    else
        ChebyshevTransformPlan{T,1,kindtuple(FIRSTKIND,N,dims...)}(FFTW.plan_r2r(x, FFTW.REDFT10, dims...; kws...))
    end
end
function plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{2}, dims...; kws...) where {T<:fftwNumber,N}
    length(x) ≤ 1 && throw(ArgumentError("Vector must contain at least 2 entries"))
    ChebyshevTransformPlan{T,2,kindtuple(SECONDKIND,N,dims...)}(FFTW.plan_r2r(x, FFTW.REDFT00, dims...; kws...))
end

plan_chebyshevtransform!(x::AbstractArray, dims...; kws...) = plan_chebyshevtransform!(x, Val(1), dims...; kws...)
plan_chebyshevtransform(x::AbstractArray, dims...; kws...) = plan_chebyshevtransform(x, Val(1), dims...; kws...)


# convert x if necessary
_plan_mul!(y::AbstractArray{T}, P::Plan{T}, x::StridedArray{T}) where T = mul!(y, P, x)
_plan_mul!(y::AbstractArray{T}, P::Plan{T}, x::AbstractArray) where T = mul!(y, P, convert(Array{T}, x))

_cheb1_rescale!(_, y::AbstractVector) = (y[1] /= 2; ldiv!(length(y), y))

function _cheb1_rescale!(d::Number, y::AbstractMatrix{T}) where T
    if isone(d)
        ldiv!(2, view(y,1,:))
    else
        ldiv!(2, view(y,:,1))
    end
    ldiv!(size(y,d), y)
end

# TODO: higher dimensional arrays
function _cheb1_rescale!(d::UnitRange, y::AbstractMatrix{T}) where T
    @assert d == 1:2
    ldiv!(2, view(y,1,:))
    ldiv!(2, view(y,:,1))
    ldiv!(prod(size(y)), y)
end

function *(P::ChebyshevTransformPlan{T,1,K,true}, x::AbstractArray{T}) where {T,K}
    n = length(x)
    n == 0 && return x

    y = P.plan*x # will be  === x if in-place
    _cheb1_rescale!(P.plan.region, y)
end

function mul!(y::AbstractArray{T}, P::ChebyshevTransformPlan{T,1,K,false}, x::AbstractArray) where {T,K}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))
    n == 0 && return y
    _plan_mul!(y, P.plan, x)
    _cheb1_rescale!(P.plan.region, y)
end


_cheb2_rescale!(_, y::AbstractVector) = (y[1] /= 2; y[end] /= 2; ldiv!(length(y)-1, y))

function _cheb2_rescale!(d::Number, y::AbstractMatrix{T}) where T
    if isone(d)
        ldiv!(2, @view(y[1,:]))
        ldiv!(2, @view(y[end,:]))
    else
        ldiv!(2, @view(y[:,1]))
        ldiv!(2, @view(y[:,end]))
    end
    ldiv!(size(y,d)-1, y)
end

# TODO: higher dimensional arrays
function _cheb2_rescale!(d::UnitRange, y::AbstractMatrix{T}) where T
    @assert d == 1:2
    ldiv!(2, @view(y[1,:]))
    ldiv!(2, @view(y[end,:]))
    ldiv!(2, @view(y[:,1]))
    ldiv!(2, @view(y[:,end]))
    ldiv!(prod(size(y) .- 1), y)
end

function *(P::ChebyshevTransformPlan{T,2,K,true}, x::AbstractArray{T}) where {T,K}
    n = length(x)
    y = P.plan*x # will be  === x if in-place
    _cheb2_rescale!(P.plan.region, y)
end

function mul!(y::AbstractArray{T}, P::ChebyshevTransformPlan{T,2,K,false}, x::AbstractArray) where {T,K}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))
    _plan_mul!(y, P.plan, x)
    _cheb2_rescale!(P.plan.region, y)
end

*(P::ChebyshevTransformPlan{T,kind,K,false}, x::AbstractArray{T}) where {T,kind,K} =
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


const IFIRSTKIND = 4

struct IChebyshevTransformPlan{T,kind,K,inplace,N,R} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,K,inplace,N,R}
    IChebyshevTransformPlan{T,kind,K,inplace,N,R}(plan) where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}(plan)
    IChebyshevTransformPlan{T,kind,K,inplace,N,R}() where {T,kind,K,inplace,N,R} = new{T,kind,K,inplace,N,R}()
end

IChebyshevTransformPlan{T,kind,K}(F::FFTW.r2rFFTWPlan{T,K,inplace,N,R}) where {T,kind,K,inplace,N,R} =
    IChebyshevTransformPlan{T,kind,K,inplace,N,R}(F)

size(P::IChebyshevTransformPlan) = isdefined(P, :plan) ? size(P.plan) : (0,)
length(P::IChebyshevTransformPlan) = isdefined(P, :plan) ? length(P.plan) : 0


# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
inv(P::ChebyshevTransformPlan{T,2,K}) where {T,K} = IChebyshevTransformPlan{T,2,K}(P.plan)
inv(P::IChebyshevTransformPlan{T,2,K}) where {T,K} = ChebyshevTransformPlan{T,2,K}(P.plan)


\(P::ChebyshevTransformPlan, x::AbstractArray) = inv(P) * x
\(P::IChebyshevTransformPlan, x::AbstractArray) = inv(P) * x


function plan_ichebyshevtransform!(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        IChebyshevTransformPlan{T,1,kindtuple(IFIRSTKIND,N,dims...),true,N,isempty(dims) ? UnitRange{Int} : typeof(dims)}()
    else
        IChebyshevTransformPlan{T,1,kindtuple(IFIRSTKIND,N,dims...)}(FFTW.plan_r2r!(x, FFTW.REDFT01, dims...; kws...))
    end
end

function plan_ichebyshevtransform!(x::AbstractArray{T}, ::Val{2}, dims...; kws...) where T<:fftwNumber
    inv(plan_chebyshevtransform!(x, Val(2), dims...; kws...))
end

function plan_ichebyshevtransform(x::AbstractArray{T,N}, ::Val{1}, dims...; kws...) where {T<:fftwNumber,N}
    if isempty(x)
        IChebyshevTransformPlan{T,1,kindtuple(IFIRSTKIND,N,dims...),false,N,isempty(dims) ? UnitRange{Int} : typeof(dims)}()
    else
        IChebyshevTransformPlan{T,1,kindtuple(IFIRSTKIND,N,dims...)}(FFTW.plan_r2r(x, FFTW.REDFT01, dims...; kws...))
    end
end

function plan_ichebyshevtransform(x::AbstractArray{T}, ::Val{2}, dims...; kws...) where T<:fftwNumber
    inv(plan_chebyshevtransform(x, Val(2), dims...; kws...))
end

plan_ichebyshevtransform!(x::AbstractArray, dims...; kws...) = plan_ichebyshevtransform!(x, Val(1), dims...; kws...)
plan_ichebyshevtransform(x::AbstractArray, dims...; kws...) = plan_ichebyshevtransform(x, Val(1), dims...; kws...)

_icheb1_prerescale!(_, x::AbstractVector) = (x[1] *= 2)
_icheb1_postrescale!(_, x::AbstractVector) = (x[1] /= 2)
function _icheb1_prerescale!(d::Number, x::AbstractVector)
    lmul!(2, isone(d) ? view(x,:,1) : view(x,1,:))
    x
end
function _icheb1_postrescale!(_, x::AbstractVector)
    ldiv(2, isone(d) ? view(x,:,1) : view(x,1,:))
    x
end

function *(P::IChebyshevTransformPlan{T,1,K,true}, x::AbstractVector{T}) where {T<:fftwNumber,K}
    n = length(x)
    n == 0 && return x

    _icheb1_prerescale!(P.plan.region, x)
    x = ldiv!(2^length(P.plan.region), P.plan*x)
    x
end

function mul!(y::AbstractVector{T}, P::IChebyshevTransformPlan{T,1,K,false}, x::AbstractVector{T}) where {T<:fftwNumber,K}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))
    n == 0 && return y

    _icheb1_prerescale!(P.plan.region, x) # Todo: don't mutate x
    _plan_mul!(y, P.plan, x)
    _icheb1_postrescale!(P.plan.region, x)
    ldiv!(2^length(P.plan.region), y)
end

_icheb2_prerescale!(_, x::AbstractVector) = (x[1] *= 2; x[end] *= 2)
_icheb2_postrescale!(_, x::AbstractVector) = (x[1] /= 2; x[end] /= 2)
function _icheb2_rescale!(d, y::AbstractVector)
    _icheb2_prerescale!(d, y)
    lmul!(convert(T, prod(size(y) .- 1))/2, y)
    y
end
function _icheb2_prerescale!(d::Number, x::AbstractVector)
    lmul!(2, isone(d) ? view(x,:,1) : view(x,1,:))
    x
end
function _icheb2_postrescale!(_, x::AbstractVector)
    ldiv(2, isone(d) ? view(x,:,1) : view(x,1,:))
    x
end

function *(P::IChebyshevTransformPlan{T,2,K, true}, x::AbstractVector{T}) where {T<:fftwNumber,K}
    n = length(x)

    _icheb2_prerescale!(P.plan.region, x)
    x = inv(P)*x
    _icheb2_rescale!(P.plan.region, x)
end

function mul!(y::AbstractVector{T}, P::IChebyshevTransformPlan{T,2,K,false}, x::AbstractVector{T}) where {T<:fftwNumber,K}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("output must match dimension"))

    _icheb2_prerescale!(P.plan.region, x)
    _plan_mul!(y, inv(P), x)
    _icheb2_postrescale!(P.plan.region, x)
    _icheb2_rescale!(P.plan.region, y)
end

*(P::IChebyshevTransformPlan{T,kind,K,false},x::AbstractVector{T}) where {T,kind,K} = mul!(similar(x), P, convert(Array,x))
ichebyshevtransform!(x::AbstractArray, dims...; kwds...) = plan_ichebyshevtransform!(x, dims...; kwds...)*x
ichebyshevtransform(x, dims...; kwds...) = plan_ichebyshevtransform(x, dims...; kwds...)*x


## Chebyshev U

struct ChebyshevUTransformPlan{T,kind,inplace,P} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,P,true,1,UnitRange{Int}}
    ChebyshevUTransformPlan{T,kind,inplace,P}(plan) where {T,kind,inplace,P} = new{T,kind,inplace,P}(plan)
    ChebyshevUTransformPlan{T,kind,inplace,P}() where {T,kind,inplace,P} = new{T,kind,inplace,P}()
end

ChebyshevUTransformPlan{T,kind,inplace}(plan::FFTW.r2rFFTWPlan{T,P}) where {T,kind,inplace,P} =
    ChebyshevUTransformPlan{T,kind,inplace,P}(plan)

ChebyshevUTransformPlan{T,kind,inplace}(plan::ChebyshevUTransformPlan{T,kind,inp,P}) where {T,kind,inplace,inp,P} =
    ChebyshevUTransformPlan{T,kind,inplace,P}(plan.plan)



function plan_chebyshevutransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        ChebyshevUTransformPlan{T,1,true,(9,)}()
    else
        ChebyshevUTransformPlan{T,1,true,(9,)}(FFTW.plan_r2r!(x, FFTW.RODFT10))
    end
end
function plan_chebyshevutransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    length(x) ≤ 1 && throw(ArgumentError("Vector must contain at least 2 entries"))
    ChebyshevUTransformPlan{T,2,true,(7,)}(FFTW.plan_r2r!(x, FFTW.RODFT00))
end

function plan_chebyshevutransform(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        ChebyshevUTransformPlan{T,1,false,(9,)}()
    else
        ChebyshevUTransformPlan{T,1,false,(9,)}(FFTW.plan_r2r!(x, FFTW.RODFT10))
    end
end
function plan_chebyshevutransform(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    length(x) ≤ 1 && throw(ArgumentError("Vector must contain at least 2 entries"))
    ChebyshevUTransformPlan{T,2,false,(7,)}(FFTW.plan_r2r!(x, FFTW.RODFT00))
end

plan_chebyshevutransform!(x::AbstractVector) = plan_chebyshevutransform!(x, Val(1))
plan_chebyshevutransform(x::AbstractVector) = plan_chebyshevutransform(x, Val(1))


function *(P::ChebyshevUTransformPlan{T,1,true},x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    for k=1:n # sqrt(1-x_j^2) weight
        x[k] *= sinpi(one(T)/(2n) + (k-one(T))/n)/n
    end
    P.plan * x
end

function *(P::ChebyshevUTransformPlan{T,2,true}, x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    c = one(T)/ (n+1)
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] *= sinpi(k*c)
    end
    lmul!(c, P.plan * x)
end

chebyshevutransform!(x::AbstractVector{T}, kind=Val(1)) where {T<:fftwNumber} =
    plan_chebyshevutransform!(x, kind)*x


"""
    chebyshevutransform(x, ::Val{kind}=Val(1))

transforms from values on a Chebyshev grid of the first or second kind to Chebyshev
coefficients of the 2nd kind (Chebyshev U expansion).
"""
chebyshevutransform(x, kind=Val(1)) = chebyshevutransform!(Array(x), kind)

*(P::ChebyshevUTransformPlan{T,k,false}, x::AbstractVector{T}) where {T,k} = ChebyshevUTransformPlan{T,k,true}(P)*Array(x)

## Inverse transforms take ChebyshevU coefficients and produce values at ChebyshevU points of the first and second kinds


struct IChebyshevUTransformPlan{T,kind,inplace,P} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,P,true,1,UnitRange{Int}}
    IChebyshevUTransformPlan{T,kind,inplace,P}(plan) where {T,kind,inplace,P} = new{T,kind,inplace,P}(plan)
    IChebyshevUTransformPlan{T,kind,inplace,P}() where {T,kind,inplace,P} = new{T,kind,inplace,P}()
end

IChebyshevUTransformPlan{T,kind,inplace}(F::FFTW.r2rFFTWPlan{T,P}) where {T,kind,inplace,P} =
    IChebyshevUTransformPlan{T,kind,inplace,P}(F)

IChebyshevUTransformPlan{T,kind,true}(F::IChebyshevUTransformPlan{T,kind,false,P}) where {T,kind,P} =
    IChebyshevUTransformPlan{T,kind,true,P}(F.plan)

function plan_ichebyshevutransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        IChebyshevUTransformPlan{T,1,true,(8,)}()
    else
        IChebyshevUTransformPlan{T,1,true,(8,)}(FFTW.plan_r2r!(x, FFTW.RODFT01))
    end
end
function plan_ichebyshevutransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    length(x) ≤ 1 && throw(ArgumentError("Vector must contain at least 2 entries"))
    IChebyshevUTransformPlan{T,2,true,(7,)}(FFTW.plan_r2r!(x, FFTW.RODFT00))
end

function plan_ichebyshevutransform(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        IChebyshevUTransformPlan{T,1,false,(8,)}()
    else
        IChebyshevUTransformPlan{T,1,false,(8,)}(FFTW.plan_r2r!(x, FFTW.RODFT01))
    end
end
function plan_ichebyshevutransform(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    length(x) ≤ 1 && throw(ArgumentError("Vector must contain at least 2 entries"))
    IChebyshevUTransformPlan{T,2,false,(7,)}(FFTW.plan_r2r!(x, FFTW.RODFT00))
end

plan_ichebyshevutransform!(x::AbstractVector) = plan_ichebyshevutransform!(x, Val(1))
plan_ichebyshevutransform(x::AbstractVector) = plan_ichebyshevutransform(x, Val(1))


function *(P::IChebyshevUTransformPlan{T,1,true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    n ≤ 1 && return x

    x = P.plan * x
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= 2sinpi(one(T)/(2n) + (k-one(T))/n)
    end
    x
end



function *(P::IChebyshevUTransformPlan{T,2,true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    n ≤ 1 && return x

    c = one(T)/ (n+1)
    lmul!((n+1)/(2n+2*one(T)), x)
    x = P.plan * x
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= sinpi(k*c)
    end
    x
end

ichebyshevutransform!(x::AbstractVector{T}, kind=Val(1)) where {T<:fftwNumber} =
    plan_ichebyshevutransform!(x, kind)*x

ichebyshevutransform(x, kind=Val(1)) = ichebyshevutransform!(Array(x), kind)

*(P::IChebyshevUTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} =
    IChebyshevUTransformPlan{T,k,true}(P)*Array(x)


## Code generation for integer inputs

for func in (:chebyshevtransform,:ichebyshevtransform,:chebyshevutransform,:ichebyshevutransform)
    @eval $func(x::AbstractVector{T}, kind=Val(1)) where {T<:Integer} = $func(convert(AbstractVector{Float64},x), kind)
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

for (kind,KIND,IKIND) in ((1,FIRSTKIND,IFIRSTKIND),(2,SECONDKIND,SECONDKIND))
    @eval begin
        plan_chebyshevtransform(x::AbstractArray{T,N}, ::Val{$kind}, dims...; kws...) where {T,N} =
            ChebyshevTransformPlan{T,$KIND,false,N,Nothing}()
        plan_ichebyshevtransform(x::AbstractArray{T,N}, ::Val{$kind}, dims...; kws...) where {T,N} =
            IChebyshevTransformPlan{T,$IKIND,false,N,Nothing}()

        plan_chebyshevtransform!(x::AbstractArray{T,N}, ::Val{$kind}, dims...; kws...) where {T,N} =
            ChebyshevTransformPlan{T,$KIND,true,N,Nothing}()
        plan_ichebyshevtransform!(x::AbstractArray{T,N}, ::Val{$kind}, dims...; kws...) where {T,N} =
            IChebyshevTransformPlan{T,$IKIND,true,N,Nothing}()
    end
end

#following Chebfun's @Chebtech1/vals2coeffs.m and @Chebtech2/vals2coeffs.m
function *(P::ChebyshevTransformPlan{T,FIRSTKIND,false,1,Nothing}, x::AbstractVector{T}) where T
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


# function *(P::ChebyshevTransformPlan{T,SECONDKIND,false,Nothing}, x::AbstractVector{T}) where T
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


*(P::ChebyshevTransformPlan{T,FIRSTKIND,true,Nothing}, x::AbstractVector{T}) where T =
    copyto!(x, ChebyshevTransformPlan{T,FIRSTKIND,false,Nothing}() * x)
# *(P::ChebyshevTransformPlan{T,2,true,Nothing}, x::AbstractVector{T}) where T =
#     copyto!(x, ChebyshevTransformPlan{T,2,false,Nothing}() * x)


#following Chebfun's @Chebtech1/vals2coeffs.m and @Chebtech2/vals2coeffs.m
function *(P::IChebyshevTransformPlan{T,IFIRSTKIND,false,Nothing}, x::AbstractVector{T}) where T
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

# function *(P::IChebyshevTransformPlan{T,SECONDKIND,true,Nothing}, x::AbstractVector{T}) where T
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

*(P::IChebyshevTransformPlan{T,IFIRSTKIND,true,Nothing}, x::AbstractVector{T}) where T =
    copyto!(x, IChebyshevTransformPlan{T,IFIRSTKIND,false,Nothing}() * x)
# *(P::IChebyshevTransformPlan{T,SECONDKIND,false,Nothing}, x::AbstractVector{T}) where T =
#     IChebyshevTransformPlan{T,SECONDKIND,true,Nothing}() * copy(x)