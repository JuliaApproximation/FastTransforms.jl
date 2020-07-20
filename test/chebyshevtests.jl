using FastTransforms, Test

@testset "Chebyshev transform"  begin
    @testset "Chebyshev points" begin
        @test chebyshevpoints(10) == chebyshevpoints(Float64, 10)
        @test chebyshevpoints(10, Val(2)) == chebyshevpoints(Float64, 10, Val(2))
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            @test chebyshevpoints(T, 0) == chebyshevpoints(T, 0, Val(2)) == T[]
            @test chebyshevpoints(T, 1) == chebyshevpoints(T, 1, Val(2)) == T[0]

            n = 20
            @inferred(chebyshevpoints(T, n))
            @inferred(chebyshevpoints(T, n, Val(2)))
            @test_throws MethodError chebyshevpoints(n, Val(-1))
        end
    end

    @testset "Chebyshev first kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_1 = chebyshevpoints(T, n)
            f = exp.(p_1)
            f̌ = @inferred(chebyshevtransform(f))

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevtransform(f̌)) ≈ exp.(p_1)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = @inferred(plan_chebyshevtransform(f))
            @test @inferred(P*f) == f̌
            @test f == f̃
            P = @inferred(plan_chebyshevtransform!(f))
            @test @inferred(P*f) == f̌
            @test f == f̌
            Pi = @inferred(plan_ichebyshevtransform(f̌))
            @test @inferred(Pi*f̌) ≈ f̃
            @test f̌ == f̄
            Pi = @inferred(plan_ichebyshevtransform!(f̌))
            @test @inferred(Pi*f̌) ≈ f̃
            @test f̌ ≈ f̃

            @test chebyshevtransform(T[1]) == T[1]
            @test ichebyshevtransform(T[1]) == T[1]
            @test_throws ArgumentError chebyshevtransform(T[])
            @test_throws ArgumentError ichebyshevtransform(T[])
        end
    end
    @testset "Chebyshev second kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_2 = chebyshevpoints(T, n, Val(2))
            f = exp.(p_2)
            f̌ = @inferred(chebyshevtransform(f, Val(2)))

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevtransform(f̌, Val(2))) ≈ exp.(p_2)

            P = @inferred(plan_chebyshevtransform!(f, Val(2)))
            Pi = @inferred(plan_ichebyshevtransform!(f, Val(2)))
            @test all(@inferred(P \ copy(f)) .=== Pi * copy(f))
            @test all(@inferred(Pi \ copy(f̌)) .=== P * copy(f̌))
            @test f ≈ P \ (P*copy(f)) ≈ P * (P\copy(f)) ≈ Pi \ (Pi*copy(f)) ≈ Pi * (Pi \ copy(f))

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = @inferred(plan_chebyshevtransform(f, Val(2)))
            @test P*f == f̌
            @test f == f̃
            P = @inferred(plan_chebyshevtransform!(f, Val(2)))
            @test P*f == f̌
            @test f == f̌
            Pi = @inferred(plan_ichebyshevtransform(f̌, Val(2)))
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            Pi = @inferred(plan_ichebyshevtransform!(f̌, Val(2)))
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃

            @test_throws ArgumentError chebyshevtransform(T[1], Val(2))
            @test_throws ArgumentError ichebyshevtransform(T[1], Val(2))
            @test_throws ArgumentError chebyshevtransform(T[], Val(2))
            @test_throws ArgumentError ichebyshevtransform(T[], Val(2))
        end
    end

    @testset "Chebyshev first kind points <-> second kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_1 = chebyshevpoints(T, n)
            f = exp.(p_1)
            f̌ = chebyshevutransform(f)

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevutransform(f̌) ≈ exp.(p_1)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = plan_chebyshevutransform(f)
            @test P*f == f̌
            @test f == f̃
            P = plan_chebyshevutransform!(f)
            @test P*f == f̌
            @test f == f̌
            Pi = plan_ichebyshevutransform(f̌)
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            Pi = plan_ichebyshevutransform!(f̌)
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃

            @test chebyshevutransform(T[1]) == T[1]
            @test ichebyshevutransform(T[1]) == T[1]
            @test chebyshevutransform(T[]) == T[]
            @test ichebyshevutransform(T[]) == T[]
        end
    end

    @testset "Chebyshev second kind points <-> second kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_2 = chebyshevpoints(T, n, Val(2))[2:end-1]
            f = exp.(p_2)
            f̌ = chebyshevutransform(f, Val(2))

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-3]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevutransform(f̌, Val(2)) ≈ exp.(p_2)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = plan_chebyshevutransform(f, Val(2))
            @test P*f == f̌
            @test f == f̃
            P = plan_chebyshevutransform!(f, Val(2))
            @test P*f == f̌
            @test f == f̌
            Pi = plan_ichebyshevutransform(f̌, Val(2))
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            Pi = plan_ichebyshevutransform!(f̌, Val(2))
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃

            @test chebyshevutransform(T[1], Val(2)) == T[1]
            @test ichebyshevutransform(T[1], Val(2)) == T[1]
            @test chebyshevutransform(T[], Val(2)) == T[]
            @test ichebyshevutransform(T[], Val(2)) == T[]
        end
    end
end
