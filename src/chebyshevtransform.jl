## Transforms take values at Chebyshev points of the first and second kinds and produce Chebyshev coefficients

abstract type ChebyshevPlan{T} <: Plan{T} end

size(P::ChebyshevPlan) = isdefined(P, :plan) ? size(P.plan) : (0,)
length(P::ChebyshevPlan) = isdefined(P, :plan) ? length(P.plan) : 0


# Check whether a ChebyshevPlan is applicable to a given input array, and
# throw an informative error if not:
function assert_applicable(p::ChebyshevPlan{T}, X::StridedArray{T}) where T
    if size(X) != size(p)
        throw(ArgumentError("Chebyshev plan applied to wrong-size array"))
    end
end

struct ChebyshevTransformPlan{T,kind,inplace,P} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,P,true,1,UnitRange{Int}}
    ChebyshevTransformPlan{T,kind,inplace,P}(plan) where {T,kind,inplace,P} = new{T,kind,inplace,P}(plan)
    ChebyshevTransformPlan{T,kind,inplace,P}() where {T,kind,inplace,P} = new{T,kind,inplace,P}()
end

ChebyshevTransformPlan{T,kind,inplace}(plan::FFTW.r2rFFTWPlan{T,P}) where {T,kind,inplace,P} = 
    ChebyshevTransformPlan{T,kind,inplace,P}(plan)

ChebyshevTransformPlan{T,kind,inplace}(plan::ChebyshevTransformPlan{T,kind,inp,P}) where {T,kind,inplace,inp,P} = 
    ChebyshevTransformPlan{T,kind,inplace,P}(plan.plan)

function plan_chebyshevtransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        ChebyshevTransformPlan{T,1,true,(5,)}()
    else
        ChebyshevTransformPlan{T,1,true,(5,)}(FFTW.plan_r2r!(x, FFTW.REDFT10))
    end
end
function plan_chebyshevtransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    length(x) ≤ 1 && throw(ArgumentError("Vector must contain at least 2 entries"))
    ChebyshevTransformPlan{T,2,true,(3,)}(FFTW.plan_r2r!(x, FFTW.REDFT00))
end


function plan_chebyshevtransform(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        ChebyshevTransformPlan{T,1,false,(5,)}()
    else
        ChebyshevTransformPlan{T,1,false,(5,)}(FFTW.plan_r2r!(x, FFTW.REDFT10))
    end
end
function plan_chebyshevtransform(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    length(x) ≤ 1 && throw(ArgumentError("Vector must contain at least 2 entries"))
    ChebyshevTransformPlan{T,2,false,(3,)}(FFTW.plan_r2r!(x, FFTW.REDFT00))
end

plan_chebyshevtransform!(x::AbstractVector) = plan_chebyshevtransform!(x, Val(1))
plan_chebyshevtransform(x::AbstractVector) = plan_chebyshevtransform(x, Val(1))


function *(P::ChebyshevTransformPlan{T,1,true}, x::AbstractVector{T}) where T
    n = length(x)
    assert_applicable(P, x)
    n == 0 && return x

    y = P.plan*x # will be  === x if in-place
    y[1] /= 2
    lmul!(inv(convert(T,n)), y)
end

function *(P::ChebyshevTransformPlan{T,2,true}, x::AbstractVector{T}) where T
    n = length(x)
    y = P.plan*x # will be  === x if in-place
    y[1] /= 2; y[end] /= 2
    lmul!(inv(convert(T,n-1)),y)
end

*(P::ChebyshevTransformPlan{T,k,false}, x::AbstractVector{T}) where {T,k} = 
    ChebyshevTransformPlan{T,k,true}(P)*copy(x)

chebyshevtransform!(x::AbstractVector{T}, kind=Val(1)) where T<:fftwNumber =
    plan_chebyshevtransform!(x, kind)*x


"""
    chebyshevtransform(x, kind=Val(1))

transforms from values on a Chebyshev grid of the first or second kind to Chebyshev
coefficients.
"""
chebyshevtransform(x, kind=Val(1)) = chebyshevtransform!(copy(x), kind)


## Inverse transforms take Chebyshev coefficients and produce values at Chebyshev points of the first and second kinds


struct IChebyshevTransformPlan{T,kind,inplace,P} <: ChebyshevPlan{T}
    plan::FFTW.r2rFFTWPlan{T,P,true,1,UnitRange{Int}}
    IChebyshevTransformPlan{T,kind,inplace,P}(plan) where {T,kind,inplace,P} = new{T,kind,inplace,P}(plan)
    IChebyshevTransformPlan{T,kind,inplace,P}() where {T,kind,inplace,P} = new{T,kind,inplace,P}()
end

IChebyshevTransformPlan{T,kind,inplace}(F::FFTW.r2rFFTWPlan{T,P}) where {T,kind,inplace,P} = 
    IChebyshevTransformPlan{T,kind,inplace,P}(F)

IChebyshevTransformPlan{T,kind,true}(F::IChebyshevTransformPlan{T,kind,false,P}) where {T,kind,P} = 
    IChebyshevTransformPlan{T,kind,true,P}(F.plan)

size(P::IChebyshevTransformPlan) = isdefined(P, :plan) ? size(P.plan) : (0,)
length(P::IChebyshevTransformPlan) = isdefined(P, :plan) ? length(P.plan) : 0


# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
inv(P::ChebyshevTransformPlan{T,2,inp}) where {T,inp} = IChebyshevTransformPlan{T,2,inp}(P.plan)
inv(P::IChebyshevTransformPlan{T,2,inp}) where {T,inp} = ChebyshevTransformPlan{T,2,inp}(P.plan)


\(P::ChebyshevTransformPlan, x::AbstractArray) = inv(P) * x
\(P::IChebyshevTransformPlan, x::AbstractArray) = inv(P) * x


function plan_ichebyshevtransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        IChebyshevTransformPlan{T,1,true,(4,)}()
    else
        IChebyshevTransformPlan{T,1,true,(4,)}(FFTW.plan_r2r!(x, FFTW.REDFT01))
    end
end

function plan_ichebyshevtransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    inv(plan_chebyshevtransform!(x, Val(2)))
end

function plan_ichebyshevtransform(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    if isempty(x)
        IChebyshevTransformPlan{T,1,false,(4,)}()
    else
        IChebyshevTransformPlan{T,1,false,(4,)}(FFTW.plan_r2r!(x, FFTW.REDFT01))
    end
end

function plan_ichebyshevtransform(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    inv(plan_chebyshevtransform(x, Val(2)))
end

plan_ichebyshevtransform!(x::AbstractVector) = plan_ichebyshevtransform!(x, Val(1))
plan_ichebyshevtransform(x::AbstractVector) = plan_ichebyshevtransform(x, Val(1))


function *(P::IChebyshevTransformPlan{T,1,true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    assert_applicable(P, x)
    n == 0 && return x

    x[1] *= 2
    x = lmul!(convert(T,0.5), P.plan*x)
    x
end 
function *(P::IChebyshevTransformPlan{T,2, true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    assert_applicable(P, x)

    x[1] *= 2; x[end] *= 2
    x = ChebyshevTransformPlan{T,2,true}(P.plan)*x
    x[1] *= 2; x[end] *= 2
    lmul!(convert(T,0.5(n-1)),x)
end

*(P::IChebyshevTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} = 
    IChebyshevTransformPlan{T,k,true}(P)*copy(x)

ichebyshevtransform!(x::AbstractVector{T}, kind=Val(1)) where {T<:fftwNumber} =
    plan_ichebyshevtransform!(x, kind)*x

ichebyshevtransform(x, kind=Val(1)) = ichebyshevtransform!(copy(x), kind)

# Matrix inputs
#
#
function chebyshevtransform!(X::AbstractMatrix{T}, ::Val{1}) where T<:fftwNumber
    if size(X) == (1,1)
        X
    else
        X=FFTW.r2r!(X,FFTW.REDFT10)
        X[:,1]/=2;X[1,:]/=2;
        lmul!(1/(size(X,1)*size(X,2)),X)
    end
end

function chebyshevtransform!(X::AbstractMatrix{T}, ::Val{2}) where T<:fftwNumber
    if size(X,1) < 2 || size(X,2) < 2
        throw(ArgumentError("Chebyshev plan applied to wrong-size array"))
    else
        X=FFTW.r2r!(X,FFTW.REDFT00)
        lmul!(1/((size(X,1)-1)*(size(X,2)-1)),X)
        X[:,1]/=2;X[:,end]/=2
        X[1,:]/=2;X[end,:]/=2
        X
    end
end
#
function ichebyshevtransform!(X::AbstractMatrix{T}, ::Val{1}) where T<:fftwNumber
    if size(X) == (1,1)
        X
    else
        X[1,:]*=2;X[:,1]*=2
        X = FFTW.r2r(X,FFTW.REDFT01)
        lmul!(0.25, X)
    end
end
function ichebyshevtransform!(X::AbstractMatrix{T}, ::Val{2}) where T<:fftwNumber
    if size(X,1) < 2 || size(X,2) < 2
        throw(ArgumentError("Chebyshev plan applied to wrong-size array"))
    else
        X[1,:]*=2;X[end,:]*=2;X[:,1]*=2;X[:,end]*=2
        X=chebyshevtransform!(X, Val(2))
        X[1,:]*=2;X[end,:]*=2;X[:,1]*=2;X[:,end]*=2
        lmul!((size(X,1)-1)*(size(X,2)-1)/4,X)
    end
end
#


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
    assert_applicable(P, x)
    n ≤ 1 && return x

    for k=1:n # sqrt(1-x_j^2) weight
        x[k] *= sinpi(one(T)/(2n) + (k-one(T))/n)/n
    end
    P.plan * x
end

function *(P::ChebyshevUTransformPlan{T,2,true}, x::AbstractVector{T}) where T
    n = length(x)
    assert_applicable(P, x)
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
chebyshevutransform(x, kind=Val(1)) = chebyshevutransform!(copy(x), kind)

*(P::ChebyshevUTransformPlan{T,k,false}, x::AbstractVector{T}) where {T,k} = ChebyshevUTransformPlan{T,k,true}(P)*copy(x)

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
    assert_applicable(P, x)
    n ≤ 1 && return x

    x = P.plan * x
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= 2sinpi(one(T)/(2n) + (k-one(T))/n)
    end
    x
end



function *(P::IChebyshevUTransformPlan{T,2,true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    assert_applicable(P, x)
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

ichebyshevutransform(x, kind=Val(1)) = ichebyshevutransform!(copy(x), kind)

*(P::IChebyshevUTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} = 
    IChebyshevUTransformPlan{T,k,true}(P)*copy(x)


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
# Use `Nothing` and fall back too FFT
###

plan_chebyshevtransform(x::AbstractVector{T}, ::Val{kind}) where {T,kind} =
    ChebyshevTransformPlan{T,kind,false,Nothing}()
plan_ichebyshevtransform(x::AbstractVector{T}, ::Val{kind}) where {T,kind} =
    IChebyshevTransformPlan{T,kind,false,Nothing}()

#following Chebfun's @Chebtech1/vals2coeffs.m and @Chebtech2/vals2coeffs.m
function *(P::ChebyshevTransformPlan{T,1,false,Nothing}, x::AbstractVector{T}) where T
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

function *(P::ChebyshevTransformPlan{T,2,false,Nothing}, x::AbstractVector{T}) where T
    n = length(x)
    if n == 1
        x
    else
        ret = ifft([x;x[end:-1:2]])[1:n]
        ret = T<:Real ? real(ret) : ret
        ret[2:n-1] *= 2
        ret
    end
end

#following Chebfun's @Chebtech1/vals2coeffs.m and @Chebtech2/vals2coeffs.m
function *(P::IChebyshevTransformPlan{T,1,false,Nothing}, x::AbstractVector{T}) where T
    n = length(x)
    if n == 1
        x
    else
        w = [exp(-im*convert(T,π)*k/2n)/2 for k=0:2n-1]
        w[1] *= 2;w[n+1] *= 0;w[n+2:end] *= -1
        ret = fft(w.*[x;one(T);x[end:-1:2]])
        ret = T<:Real ? real(ret) : ret
    end
end
function *(P::IChebyshevTransformPlan{T,2,false,Nothing}, x::AbstractVector{T}) where T
    n = length(x)
    if n == 1
        x
    else
        ##TODO: make thread safe
        x[1] *= 2;x[end] *= 2
        ret = chebyshevtransform(x, Val(2))
        x[1] /=2;x[end] /=2
        ret[1] *= 2;ret[end] *= 2
        ret *= .5*(n-1)
        ret
    end
end
