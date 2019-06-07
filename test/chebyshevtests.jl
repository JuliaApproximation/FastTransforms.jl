using FastTransforms, Compat, Compat.Test

@testset "Chebyshev transform"  begin
    @testset "Chebyshev points" begin
        @test chebyshevpoints(10; kind=1) == chebyshevpoints(Float64, 10; kind=1)
        @test chebyshevpoints(10; kind=2) == chebyshevpoints(Float64, 10; kind=2)
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            @test chebyshevpoints(T, 0, kind=1) == chebyshevpoints(T, 0, kind=2) == T[]
            @test chebyshevpoints(T, 1, kind=1) == chebyshevpoints(T, 1, kind=2) == T[0]

            n = 20
            @inferred(chebyshevpoints(T, n, kind=1))
            @inferred(chebyshevpoints(T, n, kind=2))
            @test_throws ArgumentError chebyshevpoints(n, kind=-1)
        end
    end

    @testset "Chebyshev first kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_1 = chebyshevpoints(T, n, kind=1)
            f = exp.(p_1)
            f̌ = chebyshevtransform(f; kind=1)

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevtransform(f̌; kind=1) ≈ exp.(p_1)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = plan_chebyshevtransform(f; kind=1)
            @test P*f == f̌
            @test f == f̃
            P = plan_chebyshevtransform!(f; kind=1)
            @test P*f == f̌
            @test f == f̌
            Pi = plan_ichebyshevtransform(f̌; kind=1)
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            Pi = plan_ichebyshevtransform!(f̌; kind=1)
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃

            @test chebyshevtransform(T[1]; kind=1) == T[1]
            @test ichebyshevtransform(T[1]; kind=1) == T[1]
            @test chebyshevtransform(T[]; kind=1) == T[]
            @test ichebyshevtransform(T[]; kind=1) == T[]
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

            P = plan_chebyshevtransform!(f; kind=2)
            Pi = plan_ichebyshevtransform!(f; kind=2)
            @test all((P \ copy(f)) .=== Pi * copy(f))
            @test all((Pi \ copy(f̌)) .=== P * copy(f̌))
            @test f ≈ P \ (P*copy(f)) ≈ P * (P\copy(f)) ≈ Pi \ (Pi*copy(f)) ≈ Pi * (Pi \ copy(f))

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = plan_chebyshevtransform(f; kind=2)
            @test P*f == f̌
            @test f == f̃
            P = plan_chebyshevtransform!(f; kind=2)
            @test P*f == f̌
            @test f == f̌
            Pi = plan_ichebyshevtransform(f̌; kind=2)
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            Pi = plan_ichebyshevtransform!(f̌; kind=2)
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃

            @test chebyshevtransform(T[1]; kind=2) == T[1]
            @test ichebyshevtransform(T[1]; kind=2) == T[1]
            @test chebyshevtransform(T[]; kind=2) == T[]
            @test ichebyshevtransform(T[]; kind=2) == T[]
        end
    end

    @testset "Chebyshev first kind points <-> second kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_1 = chebyshevpoints(T, n, kind=1)
            f = exp.(p_1)
            f̌ = chebyshevutransform(f; kind=1)

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevutransform(f̌; kind=1) ≈ exp.(p_1)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = plan_chebyshevutransform(f; kind=1)
            @test P*f == f̌
            @test f == f̃
            P = plan_chebyshevutransform!(f; kind=1)
            @test P*f == f̌
            @test f == f̌
            Pi = plan_ichebyshevutransform(f̌; kind=1)
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            Pi = plan_ichebyshevutransform!(f̌; kind=1)
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃

            @test chebyshevutransform(T[1]; kind=1) == T[1]
            @test ichebyshevutransform(T[1]; kind=1) == T[1]
            @test chebyshevutransform(T[]; kind=1) == T[]
            @test ichebyshevutransform(T[]; kind=1) == T[]
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
            @test ichebyshevutransform(f̌; kind=2) ≈ exp.(p_2)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = plan_chebyshevutransform(f; kind=2)
            @test P*f == f̌
            @test f == f̃
            P = plan_chebyshevutransform!(f; kind=2)
            @test P*f == f̌
            @test f == f̌
            Pi = plan_ichebyshevutransform(f̌; kind=2)
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            Pi = plan_ichebyshevutransform!(f̌; kind=2)
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃

            @test chebyshevutransform(T[1]; kind=2) == T[1]
            @test ichebyshevutransform(T[1]; kind=2) == T[1]
            @test chebyshevutransform(T[]; kind=2) == T[]
            @test ichebyshevutransform(T[]; kind=2) == T[]
        end
    end
end
