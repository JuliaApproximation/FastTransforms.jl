using FastTransforms, Test

@testset "Chebyshev transform"  begin
    @testset "Chebyshev points" begin
        @test @inferred(chebyshevpoints(10)) == @inferred(chebyshevpoints(Float64, 10))
        @test @inferred(chebyshevpoints(10, Val(2))) == @inferred(chebyshevpoints(Float64, 10, Val(2)))
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            @test chebyshevpoints(T, 0) == T[]
            @test chebyshevpoints(T, 1) == T[0]

            n = 20
            @test @inferred(chebyshevpoints(T, n)) == [sinpi(convert(T,n-2k+1)/(2n)) for k=1:n]
            @test @inferred(chebyshevpoints(T, n, Val(2))) == [sinpi(convert(T,n-2k+1)/(2n-2)) for k=1:n]

            @test_throws MethodError chebyshevpoints(n, Val(-1))
            @test_throws ArgumentError chebyshevpoints(T, 0, Val(2))
            @test_throws ArgumentError chebyshevpoints(T, 1, Val(2))
        end
    end

    @testset "Chebyshev first kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_1 = chebyshevpoints(T, n)
            f = exp.(p_1)
            f̌ = @inferred(chebyshevtransform(f))
            @test f̌ == chebyshevtransform!(copy(f))

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevtransform(f̌)) ≈ ichebyshevtransform!(copy(f̌)) ≈ exp.(p_1)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = @inferred(plan_chebyshevtransform(f))
            @test @inferred(P*f) == f̌
            @test f == f̃
            @test_throws ArgumentError P * T[1,2]
            P = @inferred(plan_chebyshevtransform!(f))
            @test @inferred(P*f) == f̌
            @test f == f̌
            @test_throws ArgumentError P * T[1,2]
            Pi = @inferred(plan_ichebyshevtransform(f̌))
            @test @inferred(Pi*f̌) ≈ f̃
            @test f̌ == f̄
            @test_throws ArgumentError Pi * T[1,2]
            Pi = @inferred(plan_ichebyshevtransform!(f̌))
            @test @inferred(Pi*f̌) ≈ f̃
            @test f̌ ≈ f̃
            @test_throws ArgumentError Pi * T[1,2]

            @test chebyshevtransform(T[1]) == T[1]
            @test ichebyshevtransform(T[1]) == T[1]
            @test chebyshevtransform(T[]) == T[]
            @test ichebyshevtransform(T[]) == T[]
        end
    end
    @testset "Chebyshev second kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_2 = chebyshevpoints(T, n, Val(2))
            f = exp.(p_2)
            f̌ = @inferred(chebyshevtransform(f, Val(2)))
            @test f̌ == chebyshevtransform!(copy(f), Val(2))

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevtransform(f̌, Val(2))) ≈ ichebyshevtransform!(copy(f̌), Val(2)) ≈ exp.(p_2)

            P = @inferred(plan_chebyshevtransform!(f, Val(2)))
            Pi = @inferred(plan_ichebyshevtransform!(f, Val(2)))
            @test all(@inferred(P \ copy(f)) .=== Pi * copy(f))
            @test all(@inferred(Pi \ copy(f̌)) .=== P * copy(f̌))
            @test f ≈ P \ (P*copy(f)) ≈ P * (P\copy(f)) ≈ Pi \ (Pi*copy(f)) ≈ Pi * (Pi \ copy(f))

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = @inferred(plan_chebyshevtransform(f, Val(2)))
            @test_throws ArgumentError P * T[1,2]
            @test P*f == f̌
            @test f == f̃
            P = @inferred(plan_chebyshevtransform!(f, Val(2)))
            @test P*f == f̌
            @test f == f̌
            @test_throws ArgumentError P * T[1,2]
            Pi = @inferred(plan_ichebyshevtransform(f̌, Val(2)))
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            @test_throws ArgumentError Pi * T[1,2]
            Pi = @inferred(plan_ichebyshevtransform!(f̌, Val(2)))
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃
            @test_throws ArgumentError Pi * T[1,2]

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
            f̌ = @inferred(chebyshevutransform(f))

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-1]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevutransform(f̌) ≈ exp.(p_1)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = @inferred(plan_chebyshevutransform(f))
            @test P*f == f̌
            @test f == f̃
            @test_throws ArgumentError P * T[1,2]
            P = @inferred(plan_chebyshevutransform!(f))
            @test P*f == f̌
            @test f == f̌
            @test_throws ArgumentError P * T[1,2]
            Pi = @inferred(plan_ichebyshevutransform(f̌))
            @test Pi*f̌ ≈ f̃
            @test f̌ == f̄
            @test_throws ArgumentError Pi * T[1,2]
            Pi = @inferred(plan_ichebyshevutransform!(f̌))
            @test Pi*f̌ ≈ f̃
            @test f̌ ≈ f̃
            @test_throws ArgumentError Pi * T[1,2]

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
            f̌ = @inferred(chebyshevutransform(f, Val(2)))

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-3]' * f̌
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevutransform(f̌, Val(2))) ≈ exp.(p_2)

            f̃ = copy(f)
            f̄ = copy(f̌)
            P = @inferred(plan_chebyshevutransform(f, Val(2)))
            @test @inferred(P*f) == f̌
            @test f == f̃
            @test_throws ArgumentError P * T[1,2]
            P = @inferred(plan_chebyshevutransform!(f, Val(2)))
            @test @inferred(P*f) == f̌
            @test f == f̌
            @test_throws ArgumentError P * T[1,2]
            Pi = @inferred(plan_ichebyshevutransform(f̌, Val(2)))
            @test @inferred(Pi*f̌) ≈ f̃
            @test f̌ == f̄
            @test_throws ArgumentError Pi * T[1,2]
            Pi = @inferred(plan_ichebyshevutransform!(f̌, Val(2)))
            @test @inferred(Pi*f̌) ≈ f̃
            @test f̌ ≈ f̃
            @test_throws ArgumentError Pi * T[1,2]

            @test_throws ArgumentError chebyshevutransform(T[1], Val(2))
            @test_throws ArgumentError ichebyshevutransform(T[1], Val(2))
            @test_throws ArgumentError chebyshevutransform(T[], Val(2))
            @test_throws ArgumentError ichebyshevutransform(T[], Val(2))
        end
    end

    @testset "matrix" begin
        X = randn(4,5)
        @test @inferred(chebyshevtransform(X,1)) ≈ @inferred(chebyshevtransform!(copy(X),1)) ≈ hcat(chebyshevtransform.([X[:,k] for k=axes(X,2)])...)
        @test chebyshevtransform(X,2) ≈ chebyshevtransform!(copy(X),2) ≈ hcat(chebyshevtransform.([X[k,:] for k=axes(X,1)])...)'

        @test @inferred(chebyshevtransform(X)) == @inferred(chebyshevtransform!(copy(X))) == chebyshevtransform(chebyshevtransform(X,1),2)


        X = randn(1,1)
        @test chebyshevtransform!(copy(X), Val(1)) == ichebyshevtransform!(copy(X), Val(1)) == X
        @test_throws ArgumentError chebyshevtransform!(copy(X), Val(2))
        @test_throws ArgumentError ichebyshevtransform!(copy(X), Val(2))

        X = randn(10,11)
        
        # manual 2D Chebyshev
        X̌ = copy(X)
        for j in axes(X̌,2)
            chebyshevtransform!(view(X̌,:,j))
        end
        for k in axes(X̌,1)
            chebyshevtransform!(view(X̌,k,:))
        end
        @test chebyshevtransform!(copy(X), Val(1)) ≈ X̌
        @test ichebyshevtransform!(copy(X̌), Val(1)) ≈ X

        # manual 2D Chebyshev
        X̌ = copy(X)
        for j in axes(X̌,2)
            chebyshevtransform!(view(X̌,:,j), Val(2))
        end
        for k in axes(X̌,1)
            chebyshevtransform!(view(X̌,k,:), Val(2))
        end
        @test chebyshevtransform!(copy(X), Val(2)) ≈ X̌
        @test ichebyshevtransform!(copy(X̌), Val(2)) ≈ X
    end

    @testset "Integer" begin
        @test chebyshevtransform([1,2,3]) == chebyshevtransform([1.,2,3])
        @test chebyshevtransform([1,2,3], Val(2)) == chebyshevtransform([1.,2,3], Val(2))
        @test ichebyshevtransform([1,2,3]) == ichebyshevtransform([1.,2,3])
        @test ichebyshevtransform([1,2,3], Val(2)) == ichebyshevtransform([1.,2,3], Val(2))
        
        @test chebyshevutransform([1,2,3]) == chebyshevutransform([1.,2,3])
        @test chebyshevutransform([1,2,3], Val(2)) == chebyshevutransform([1.,2,3], Val(2))
        @test ichebyshevutransform([1,2,3]) == ichebyshevutransform([1.,2,3])
        @test ichebyshevutransform([1,2,3], Val(2)) == ichebyshevutransform([1.,2,3], Val(2))
    end

    @testset "BigFloat" begin
        x = BigFloat[1,2,3]
        @test ichebyshevtransform(chebyshevtransform(x)) ≈ x
        @test plan_chebyshevtransform(x)x ≈ chebyshevtransform(x)
        @test plan_ichebyshevtransform(x)x ≈ ichebyshevtransform(x)
        @test plan_chebyshevtransform!(x)copy(x) ≈ chebyshevtransform(x)
        @test plan_ichebyshevtransform!(x)copy(x) ≈ ichebyshevtransform(x)
    end

    @testset "immutable vectors" begin
        F = plan_chebyshevtransform([1.,2,3])
        @test chebyshevtransform(1.0:3) == F * (1:3)
    end
end
