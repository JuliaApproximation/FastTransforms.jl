using FastTransforms, Compat, Compat.Test

@testset "Chebyshev transform"  begin
    @testset "Chebyshev first kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_1 = chebyshevpoints(T, n, kind=1)
            f = exp.(p_1)
            f̌ = chebyshevtransform(f; kind=1)

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevtransform(f̌; kind=1) ≈ exp.(p_1)
        end
    end
    @testset "Chebyshev second kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_2 = chebyshevpoints(T, n, kind=2)
            f = exp.(p_2)
            f̌ = chebyshevtransform(f; kind=2)

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevtransform(f̌; kind=2) ≈ exp.(p_2)
        end
    end

    @testset "Chebyshev second kind points <-> second kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_2 = chebyshevpoints(T, n, kind=2)[2:end-1]
            f = exp.(p_2)
            f̌ = chebyshevutransform(f; kind=2)

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-3]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            # @test ichebyshevtransform(f̌; kind=2) ≈ exp.(p_2)
        end
    end
end
