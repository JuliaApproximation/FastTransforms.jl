UpperTriangularHierarchicalMatrix{T}(::Type{T}, f::Function, bd::Int64) = UpperTriangularHierarchicalMatrix(T, f, bd, bd)
UpperTriangularHierarchicalMatrix{T}(::Type{T}, f::Function, b::Int64, d::Int64) = UpperTriangularHierarchicalMatrix(T, f, 1, b, 1, d)

function UpperTriangularHierarchicalMatrix{T}(::Type{T}, f::Function, a::Int64, b::Int64, c::Int64, d::Int64)
    if (b-a+1) < BLOCKSIZE(T) && (d-c+1) < BLOCKSIZE(T)
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = T[j ≥ i ? f(T,i,j) : zero(T) for i=a:a+i, j=c:c+j]
        H[Block(1), Block(2)] = T[f(T,i,j) for i=a:a+i, j=c+j+1:d]
        H[Block(2), Block(2)] = T[j ≥ i ? f(T,i,j) : zero(T) for i=a+i+1:b, j=c+j+1:d]

        H
    else
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = UpperTriangularHierarchicalMatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = HierarchicalMatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(2)] = UpperTriangularHierarchicalMatrix(T, f, a+i+1, b, c+j+1, d)

        H
    end
end

function HierarchicalMatrix{T}(::Type{T}, f::Function, a::Int64, b::Int64, c::Int64, d::Int64)
    if (b-a+1) < BLOCKSIZE(T) && (d-c+1) < BLOCKSIZE(T)
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = barycentricmatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = T[f(T,i,j) for i=a+i+1:b, j=c:c+j]
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)

        H
    else
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = barycentricmatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = HierarchicalMatrix(T, f, a+i+1, b, c, c+j)
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)

        H
    end
end

function Meven{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x)).*Λ(1.0*(y+x-2)))
end

function Modd{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x)).*Λ(1.0*(y+x-1)))
end

function Leven{T}(::Type{T}, x, y)
    if x == y
        if x == 1.0
            T(1.0)
        else
            T(sqrt(π)/2/Λ(2.0*(x-1.0)))
        end
    else
        T(-(y-1.0)*(2.0*x-1.5)/(2.0*x+2.0*y-3.0)/(y-x)*Λ(1.0*(y-x-1.0)).*Λ(1.0*(y+x-2.5)))
    end
end

function Lodd{T}(::Type{T}, x, y)
    if x == y
        if x == 1.0
            T(1.0)
        else
            T(sqrt(π)/2/Λ(2.0*x-1.0))
        end
    else
        T(-(2.0*y-1.0)*(2.0*x-0.5)/(2.0*x+2.0*y-1.0)/(2.0*y-2.0*x)*Λ(1.0*(y-x-1)).*Λ(0.5*(2.0*y+2.0*x-3)))
    end
end

immutable LegendreToChebyshevPlan{T} <: AbstractMatrix{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
end

Base.size(P::LegendreToChebyshevPlan) = (size(P.even, 1)+size(P.odd, 1), size(P.even, 2)+size(P.odd, 2))
function Base.getindex(P::LegendreToChebyshevPlan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]
    else
        zero(eltype(P))
    end
end

immutable ChebyshevToLegendrePlan{T} <: AbstractMatrix{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
end

Base.size(P::ChebyshevToLegendrePlan) = (size(P.even, 1)+size(P.odd, 1), size(P.even, 2)+size(P.odd, 2))
function Base.getindex(P::ChebyshevToLegendrePlan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]
    else
        zero(eltype(P))
    end
end

LegendreToChebyshevPlan(v::Vector) = LegendreToChebyshevPlan(plan_even_leg2cheb(v), plan_odd_leg2cheb(v))
ChebyshevToLegendrePlan(v::Vector) = ChebyshevToLegendrePlan(plan_even_cheb2leg(v), plan_odd_cheb2leg(v))

plan_leg2cheb(v::Vector) = LegendreToChebyshevPlan(v)
plan_cheb2leg(v::Vector) = ChebyshevToLegendrePlan(v)

evenlength(v::Vector) = (L = length(v); iseven(L) ? L÷2 : (L+1)÷2)
oddlength(v::Vector) = (L = length(v); iseven(L) ? L÷2 : (L-1)÷2)

plan_even_leg2cheb(v::Vector) = UpperTriangularHierarchicalMatrix(eltype(v), Meven, evenlength(v))
plan_odd_leg2cheb(v::Vector) = UpperTriangularHierarchicalMatrix(eltype(v), Modd, oddlength(v))

plan_even_cheb2leg(v::Vector) = UpperTriangularHierarchicalMatrix(eltype(v), Leven, evenlength(v))
plan_odd_cheb2leg(v::Vector) = UpperTriangularHierarchicalMatrix(eltype(v), Lodd, oddlength(v))

function *(P::LegendreToChebyshevPlan,v::AbstractVector)
    u = zero(v)
    u[1:2:end] = P.even*view(v,1:2:length(v))
    u[2:2:end] = P.odd*view(v,2:2:length(v))
    scale!(2/π, u)
    u[1] *= 0.5
    u
end

function *(P::ChebyshevToLegendrePlan,v::AbstractVector)
    u = zero(v)
    u[1:2:end] = P.even*view(v,1:2:length(v))
    u[2:2:end] = P.odd*view(v,2:2:length(v))
    u
end

leg2cheb(v::Vector) = plan_leg2cheb(v)*v
cheb2leg(v::Vector) = plan_cheb2leg(v)*v
