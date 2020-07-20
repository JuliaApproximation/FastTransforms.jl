## Transforms take values at Chebyshev points of the first and second kinds and produce Chebyshev coefficients


struct ChebyshevTransformPlan{T,kind,inplace,P} <: Plan{T}
    plan::P
end

ChebyshevTransformPlan{k,inp}(plan) where {k,inp} =
    ChebyshevTransformPlan{eltype(plan),k,inp,typeof(plan)}(plan)



function plan_chebyshevtransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    plan = isempty(x) ? fill(one(T),1,length(x)) : FFTW.plan_r2r!(x, FFTW.REDFT10)
    ChebyshevTransformPlan{1,true}(plan)
end


function plan_chebyshevtransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    plan = length(x) ≤ 1 ? fill(one(T),1,length(x)) : FFTW.plan_r2r!(x, FFTW.REDFT00)
    ChebyshevTransformPlan{2,true}(plan)
end

plan_chebyshevtransform!(x::AbstractVector) = plan_chebyshevtransform!(x, Val(1))
function plan_chebyshevtransform(x::AbstractVector{T}, ::Val{kind}=Val(1)) where {T<:fftwNumber,kind}
    plan = plan_chebyshevtransform!(x, Val(kind))
    ChebyshevTransformPlan{kind,false}(plan)
end

function *(P::ChebyshevTransformPlan{T,1,true}, x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    x = P.plan*x
    x[1] /= 2
    lmul!(inv(convert(T,n)), x)
end

function *(P::ChebyshevTransformPlan{T,2,true}, x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    x = P.plan*x
    x[1] /= 2; x[end] /= 2
    lmul!(inv(convert(T,n-1)),x)
end

chebyshevtransform!(x::AbstractVector{T}, kind=Val(1)) where T<:fftwNumber =
    plan_chebyshevtransform!(x, kind)*x


"""
    chebyshevtransform(x, kind=Val(1))

transforms from values on a Chebyshev grid of the first or second kind to Chebyshev
coefficients.
"""
chebyshevtransform(x, kind=Val(1)) = chebyshevtransform!(copy(x), kind)

*(P::ChebyshevTransformPlan{T,k,false}, x::AbstractVector{T}) where {T,k} = P.plan*copy(x)




## Inverse transforms take Chebyshev coefficients and produce values at Chebyshev points of the first and second kinds


struct IChebyshevTransformPlan{T,kind,inplace,P}
    plan::P
end

# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
inv(P::ChebyshevTransformPlan{T,2,true}) where T = IChebyshevTransformPlan{T,2,true,typeof(P)}(P)
inv(P::IChebyshevTransformPlan{T,2,true}) where T = P.plan

\(P::ChebyshevTransformPlan, x::AbstractArray) = inv(P) * x
\(P::IChebyshevTransformPlan, x::AbstractArray) = inv(P) * x


function plan_ichebyshevtransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    plan = isempty(x) ? fill(one(T),1,length(x)) : FFTW.plan_r2r!(x, FFTW.REDFT01)
    IChebyshevTransformPlan{T,1,true,typeof(plan)}(plan)
end

function plan_ichebyshevtransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    inv(plan_chebyshevtransform!(x, Val(2)))
end

plan_ichebyshevtransform!(x::AbstractVector) = plan_ichebyshevtransform!(x, Val(1))

function plan_ichebyshevtransform(x::AbstractVector{T}, ::Val{kind}=Val(1)) where {T<:fftwNumber,kind}
    plan = plan_ichebyshevtransform!(similar(Vector{T},axes(x)), Val{kind}())
    IChebyshevTransformPlan{T,kind,false,typeof(plan)}(plan)
end

function *(P::IChebyshevTransformPlan{T,1,true}, x::AbstractVector{T}) where T<:fftwNumber
    isempty(x) && return x
    x[1] *=2
    x = lmul!(convert(T,0.5), P.plan*x)
    x
end

function *(P::IChebyshevTransformPlan{T,2,true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    n ≤ 1 && return x
    x[1] *= 2; x[end] *= 2
    x = P.plan*x
    x[1] *= 2; x[end] *= 2
    lmul!(convert(T,0.5(n-1)),x)
end

ichebyshevtransform!(x::AbstractVector{T}, kind=Val(1)) where {T<:fftwNumber} =
    plan_ichebyshevtransform!(x, kind)*x

ichebyshevtransform(x, kind=Val(1)) = ichebyshevtransform!(copy(x), kind)

*(P::IChebyshevTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} = P.plan*copy(x)

## Code generation for integer inputs

for func in (:chebyshevtransform,:ichebyshevtransform)
    @eval $func(x::AbstractVector{T}, kind=Val(1)) where {T<:Integer} = $func(convert(Float64,x), kind)
end


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
    if size(X) == (1,1)
        X
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
    if size(X) == (1,1)
        X
    else
        X[1,:]*=2;X[end,:]*=2;X[:,1]*=2;X[:,end]*=2
        X=chebyshevtransform!(X;kind=kind)
        X[1,:]*=2;X[end,:]*=2;X[:,1]*=2;X[:,end]*=2
        lmul!((size(X,1)-1)*(size(X,2)-1)/4,X)
    end
end
#


## Chebyshev U

struct ChebyshevUTransformPlan{T,kind,inplace,P} <: Plan{T}
    plan::P
end

ChebyshevUTransformPlan{k,inp}(plan) where {k,inp} =
    ChebyshevUTransformPlan{eltype(plan),k,inp,typeof(plan)}(plan)



function plan_chebyshevutransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    plan = isempty(x) ? fill(one(T),1,length(x)) : FFTW.plan_r2r!(x, FFTW.RODFT10)
    ChebyshevUTransformPlan{1,true}(plan)
end
function plan_chebyshevutransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    plan = length(x) ≤ 1 ? fill(one(T),1,length(x)) : FFTW.plan_r2r!(x, FFTW.RODFT00)
    ChebyshevUTransformPlan{2,true}(plan)
end

plan_chebyshevutransform!(x::AbstractVector) = plan_chebyshevutransform!(x, Val(1))

function plan_chebyshevutransform(x::AbstractVector{T}, ::Val{kind}=Val(1)) where {T<:fftwNumber,kind}
    plan = plan_chebyshevutransform!(x, Val{kind}())
    ChebyshevUTransformPlan{kind,false}(plan)
end

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
chebyshevutransform(x, kind=Val(1)) = chebyshevutransform!(copy(x), kind)

*(P::ChebyshevUTransformPlan{T,k,false}, x::AbstractVector{T}) where {T,k} = P.plan*copy(x)

## Inverse transforms take ChebyshevU coefficients and produce values at ChebyshevU points of the first and second kinds


struct IChebyshevUTransformPlan{T,kind,inplace,P}
    plan::P
end


function plan_ichebyshevutransform!(x::AbstractVector{T}, ::Val{1}) where T<:fftwNumber
    plan = isempty(x) ? fill(one(T),1,length(x)) : FFTW.plan_r2r!(x, FFTW.RODFT01)
    IChebyshevUTransformPlan{T,1,true,typeof(plan)}(plan)
end
function plan_ichebyshevutransform!(x::AbstractVector{T}, ::Val{2}) where T<:fftwNumber
    plan = length(x) ≤ 1 ? fill(one(T),1,length(x)) : FFTW.plan_r2r!(x, FFTW.RODFT00)
    IChebyshevUTransformPlan{T,2,true,typeof(plan)}(plan)
end

plan_ichebyshevutransform!(x::AbstractVector) = plan_ichebyshevutransform!(x, Val(1))

function plan_ichebyshevutransform(x::AbstractVector{T}, ::Val{kind}=Val(1)) where {T<:fftwNumber,kind}
    plan = plan_ichebyshevutransform!(similar(Vector{T},axes(x)), Val{kind}())
    IChebyshevUTransformPlan{T,kind,false,typeof(plan)}(plan)
end

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

ichebyshevutransform(x, kind=Val(1)) = ichebyshevutransform!(copy(x), kind)

*(P::IChebyshevUTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} = P.plan*copy(x)

## Code generation for integer inputs

for func in (:chebyshevutransform,:ichebyshevutransform)
    @eval $func(x::AbstractVector{T}, kind=Val(1)) where {T<:Integer} = $func(convert(Float64,x), kind)
end




## points

struct ChebyshevGrid{kind,T} <: AbstractVector{T}
    n::Int
end

ChebyshevGrid{kind}(n::Integer) where kind = ChebyshevGrid{kind,Float64}(n)

size(g::ChebyshevGrid) = (g.n,)
getindex(g::ChebyshevGrid{1,T}, k::Integer) where T =
    sinpi(convert(T,g.n-2k+1)/(2g.n))

function getindex(g::ChebyshevGrid{2,T}, k::Integer) where T
    g.n == 1 && return zero(T)
    sinpi(convert(T,g.n-2k+1)/(2g.n-2))
end

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
