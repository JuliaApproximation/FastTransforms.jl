function zero_spurious_modes!(A::AbstractMatrix)
    M, N = size(A)
    n = N÷2
    @inbounds for j = 1:n
        @simd for i = M-j+1:M
            A[i,2j] = 0
            A[i,2j+1] = 0
        end
    end
    A
end

include("slowplan.jl")
include("Butterfly.jl")
include("fastplan.jl")
include("thinplan.jl")
