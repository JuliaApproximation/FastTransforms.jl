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
            g = @inferred(chebyshevtransform(f))
            @test g == chebyshevtransform!(copy(f))

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * g
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevtransform(g)) ≈ ichebyshevtransform!(copy(g)) ≈ exp.(p_1)

            fcopy = copy(f)
            gcopy = copy(g)
            P = @inferred(plan_chebyshevtransform(f))
            @test @inferred(P*f) == g
            @test f == fcopy
            @test_throws ArgumentError P * T[1,2]
            P2 = @inferred(plan_chebyshevtransform(f, Val(1), 1:1))
            @test @inferred(P2*f) == g
            @test_throws ArgumentError P * T[1,2]

            P = @inferred(plan_chebyshevtransform!(f))
            @test @inferred(P*f) == g
            @test f == g
            @test_throws ArgumentError P * T[1,2]
            f .= fcopy
            P2 = @inferred(plan_chebyshevtransform!(f, 1:1))
            @test @inferred(P2*f) == g
            @test f == g
            @test_throws ArgumentError P * T[1,2]

            Pi = @inferred(plan_ichebyshevtransform(g))
            @test @inferred(Pi*g) ≈ fcopy
            @test g == gcopy
            @test_throws ArgumentError Pi * T[1,2]
            Pi2 = @inferred(plan_ichebyshevtransform(g, 1:1))
            @test @inferred(Pi2*g) ≈ fcopy
            @test g == gcopy
            @test_throws ArgumentError Pi * T[1,2]

            Pi = @inferred(plan_ichebyshevtransform!(g))
            @test @inferred(Pi*g) ≈ fcopy
            @test g ≈ fcopy
            g .= gcopy
            @test_throws ArgumentError Pi * T[1,2]
            Pi2 = @inferred(plan_ichebyshevtransform!(g, 1:1))
            @test @inferred(Pi2*g) ≈ fcopy
            @test g ≈ fcopy
            @test_throws ArgumentError Pi * T[1,2]

            v = T[1]
            @test chebyshevtransform(v) == v
            @test ichebyshevtransform(v) == v
            @test chebyshevtransform!(v) === v
            @test ichebyshevtransform!(v) === v

            v = T[]
            @test chebyshevtransform(v) == v
            @test ichebyshevtransform(v) == v
            @test chebyshevtransform!(v) === v
            @test ichebyshevtransform!(v) === v
        end
    end
    @testset "Chebyshev second kind points <-> first kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_2 = chebyshevpoints(T, n, Val(2))
            f = exp.(p_2)
            g = @inferred(chebyshevtransform(f, Val(2)))
            @test g == chebyshevtransform!(copy(f), Val(2))

            f̃ = x -> [cos(k*acos(x)) for k=0:n-1]' * g
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevtransform(g, Val(2))) ≈ ichebyshevtransform!(copy(g), Val(2)) ≈ exp.(p_2)

            P = @inferred(plan_chebyshevtransform!(f, Val(2)))
            Pi = @inferred(plan_ichebyshevtransform!(f, Val(2)))
            @test all(@inferred(P \ copy(f)) .=== Pi * copy(f))
            @test all(@inferred(Pi \ copy(g)) .=== P * copy(g))
            @test f ≈ P \ (P*copy(f)) ≈ P * (P\copy(f)) ≈ Pi \ (Pi*copy(f)) ≈ Pi * (Pi \ copy(f))

            fcopy = copy(f)
            gcopy = copy(g)

            P = @inferred(plan_chebyshevtransform(f, Val(2)))
            @test P*f == g
            @test f == fcopy
            @test_throws ArgumentError P * T[1,2]
            P = @inferred(plan_chebyshevtransform(f, Val(2), 1:1))
            @test P*f == g
            @test f == fcopy
            @test_throws ArgumentError P * T[1,2]

            P = @inferred(plan_chebyshevtransform!(f, Val(2)))
            @test P*f == g
            @test f == g
            @test_throws ArgumentError P * T[1,2]
            f .= fcopy
            P = @inferred(plan_chebyshevtransform!(f, Val(2), 1:1))
            @test P*f == g
            @test f == g
            @test_throws ArgumentError P * T[1,2]

            Pi = @inferred(plan_ichebyshevtransform(g, Val(2)))
            @test Pi*g ≈ fcopy
            @test g == gcopy
            @test_throws ArgumentError Pi * T[1,2]
            Pi = @inferred(plan_ichebyshevtransform(g, Val(2), 1:1))
            @test Pi*g ≈ fcopy
            @test g == gcopy
            @test_throws ArgumentError Pi * T[1,2]

            Pi = @inferred(plan_ichebyshevtransform!(g, Val(2)))
            @test Pi*g ≈ fcopy
            @test g ≈ fcopy
            @test_throws ArgumentError Pi * T[1,2]
            g .= gcopy
            Pi = @inferred(plan_ichebyshevtransform!(g, Val(2), 1:1))
            @test Pi*g ≈ fcopy
            @test g ≈ fcopy
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
            g = @inferred(chebyshevutransform(f))
            @test f ≈ exp.(p_1)

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-1]' * g
            @test f̃(0.1) ≈ exp(T(0.1))
            @test ichebyshevutransform(g) ≈ exp.(p_1)

            fcopy = copy(f)
            gcopy = copy(g)
            P = @inferred(plan_chebyshevutransform(f))
            @test P*f ≈ g
            @test f ≈ fcopy
            @test_throws ArgumentError P * T[1,2]
            P = @inferred(plan_chebyshevutransform(f, 1:1))
            @test P*f ≈ g
            @test f ≈ fcopy
            @test_throws ArgumentError P * T[1,2]

            P = @inferred(plan_chebyshevutransform!(f))
            @test P*f ≈ g
            @test f ≈ g
            @test_throws ArgumentError P * T[1,2]
            f .= fcopy
            P = @inferred(plan_chebyshevutransform!(f))
            @test P*f ≈ g
            @test f ≈ g
            @test_throws ArgumentError P * T[1,2]

            Pi = @inferred(plan_ichebyshevutransform(g))
            @test Pi*g ≈ fcopy
            @test g == gcopy
            @test_throws ArgumentError Pi * T[1,2]
            Pi = @inferred(plan_ichebyshevutransform(g, 1:1))
            @test Pi*g ≈ fcopy
            @test g == gcopy
            @test_throws ArgumentError Pi * T[1,2]

            Pi = @inferred(plan_ichebyshevutransform!(g))
            @test Pi*g ≈ fcopy
            @test g ≈ fcopy
            @test_throws ArgumentError Pi * T[1,2]
            g .= gcopy
            Pi = @inferred(plan_ichebyshevutransform!(g))
            @test Pi*g ≈ fcopy
            @test g ≈ fcopy
            @test_throws ArgumentError Pi * T[1,2]

            v = T[1]
            @test chebyshevutransform(v) == v
            @test ichebyshevutransform(v) == v
            @test chebyshevutransform!(v) === v
            @test ichebyshevutransform!(v) === v

            v = T[]
            @test chebyshevutransform(v) == v
            @test ichebyshevutransform(v) == v
            @test chebyshevutransform!(v) === v
            @test ichebyshevutransform!(v) === v
        end
    end
    @testset "Chebyshev second kind points <-> second kind coefficients" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            n = 20
            p_2 = chebyshevpoints(T, n, Val(2))[2:end-1]
            f = exp.(p_2)
            g = @inferred(chebyshevutransform(f, Val(2)))

            f̃ = x -> [sin((k+1)*acos(x))/sin(acos(x)) for k=0:n-3]' * g
            @test f̃(0.1) ≈ exp(T(0.1))
            @test @inferred(ichebyshevutransform(g, Val(2))) ≈ f ≈ exp.(p_2)

            fcopy = copy(f)
            gcopy = copy(g)
            P = @inferred(plan_chebyshevutransform(f, Val(2)))
            @test @inferred(P*f) ≈ g
            @test f ≈ fcopy
            @test_throws ArgumentError P * T[1,2]
            P = @inferred(plan_chebyshevutransform(f, Val(2), 1:1))
            @test @inferred(P*f) ≈ g
            @test f ≈ fcopy
            @test_throws ArgumentError P * T[1,2]

            P = @inferred(plan_chebyshevutransform!(f, Val(2)))
            @test @inferred(P*f) ≈ g
            @test f ≈ g
            @test_throws ArgumentError P * T[1,2]
            f .= fcopy
            P = @inferred(plan_chebyshevutransform!(f, Val(2), 1:1))
            @test @inferred(P*f) ≈ g
            @test f ≈ g
            @test_throws ArgumentError P * T[1,2]

            Pi = @inferred(plan_ichebyshevutransform(g, Val(2)))
            @test @inferred(Pi*g) ≈ fcopy
            @test g ≈ gcopy
            @test_throws ArgumentError Pi * T[1,2]

            Pi = @inferred(plan_ichebyshevutransform!(g, Val(2)))
            @test @inferred(Pi*g) ≈ fcopy
            @test g ≈ fcopy
            @test_throws ArgumentError Pi * T[1,2]
            g .= gcopy
            Pi = @inferred(plan_ichebyshevutransform!(g, Val(2)))
            @test @inferred(Pi*g) ≈ fcopy
            @test g ≈ fcopy
            @test_throws ArgumentError Pi * T[1,2]

            @test_throws ArgumentError chebyshevutransform(T[1], Val(2))
            @test_throws ArgumentError ichebyshevutransform(T[1], Val(2))
            @test_throws ArgumentError chebyshevutransform(T[], Val(2))
            @test_throws ArgumentError ichebyshevutransform(T[], Val(2))
        end
    end

    @testset "matrix" begin
        X = randn(4,5)
        @testset "chebyshevtransform" begin
            @test @inferred(chebyshevtransform(X,1)) ≈ @inferred(chebyshevtransform!(copy(X),1)) ≈ hcat(chebyshevtransform.([X[:,k] for k=axes(X,2)])...)
            @test chebyshevtransform(X,2) ≈ chebyshevtransform!(copy(X),2) ≈ hcat(chebyshevtransform.([X[k,:] for k=axes(X,1)])...)'
            @test @inferred(chebyshevtransform(X,Val(2),1)) ≈ @inferred(chebyshevtransform!(copy(X),Val(2),1)) ≈ hcat(chebyshevtransform.([X[:,k] for k=axes(X,2)],Val(2))...)
            @test chebyshevtransform(X,Val(2),2) ≈ chebyshevtransform!(copy(X),Val(2),2) ≈ hcat(chebyshevtransform.([X[k,:] for k=axes(X,1)],Val(2))...)'

            @test @inferred(chebyshevtransform(X)) ≈ @inferred(chebyshevtransform!(copy(X))) ≈ chebyshevtransform(chebyshevtransform(X,1),2)
            @test @inferred(chebyshevtransform(X,Val(2))) ≈ @inferred(chebyshevtransform!(copy(X),Val(2))) ≈ chebyshevtransform(chebyshevtransform(X,Val(2),1),Val(2),2)
        end

        @testset "ichebyshevtransform" begin
            @test @inferred(ichebyshevtransform(X,1)) ≈ @inferred(ichebyshevtransform!(copy(X),1)) ≈ hcat(ichebyshevtransform.([X[:,k] for k=axes(X,2)])...)
            @test ichebyshevtransform(X,2) ≈ ichebyshevtransform!(copy(X),2) ≈ hcat(ichebyshevtransform.([X[k,:] for k=axes(X,1)])...)'
            @test @inferred(ichebyshevtransform(X,Val(2),1)) ≈ @inferred(ichebyshevtransform!(copy(X),Val(2),1)) ≈ hcat(ichebyshevtransform.([X[:,k] for k=axes(X,2)],Val(2))...)
            @test ichebyshevtransform(X,Val(2),2) ≈ ichebyshevtransform!(copy(X),Val(2),2) ≈ hcat(ichebyshevtransform.([X[k,:] for k=axes(X,1)],Val(2))...)'

            @test @inferred(ichebyshevtransform(X)) ≈ @inferred(ichebyshevtransform!(copy(X))) ≈ ichebyshevtransform(ichebyshevtransform(X,1),2)
            @test @inferred(ichebyshevtransform(X,Val(2))) ≈ @inferred(ichebyshevtransform!(copy(X),Val(2))) ≈ ichebyshevtransform(ichebyshevtransform(X,Val(2),1),Val(2),2)

            @test ichebyshevtransform(chebyshevtransform(X)) ≈ X
            @test chebyshevtransform(ichebyshevtransform(X)) ≈ X
        end

        @testset "chebyshevutransform" begin
            @test @inferred(chebyshevutransform(X,1)) ≈ @inferred(chebyshevutransform!(copy(X),1)) ≈ hcat(chebyshevutransform.([X[:,k] for k=axes(X,2)])...)
            @test chebyshevutransform(X,2) ≈ chebyshevutransform!(copy(X),2) ≈ hcat(chebyshevutransform.([X[k,:] for k=axes(X,1)])...)'
            @test @inferred(chebyshevutransform(X,Val(2),1)) ≈ @inferred(chebyshevutransform!(copy(X),Val(2),1)) ≈ hcat(chebyshevutransform.([X[:,k] for k=axes(X,2)],Val(2))...)
            @test chebyshevutransform(X,Val(2),2) ≈ chebyshevutransform!(copy(X),Val(2),2) ≈ hcat(chebyshevutransform.([X[k,:] for k=axes(X,1)],Val(2))...)'

            @test @inferred(chebyshevutransform(X)) ≈ @inferred(chebyshevutransform!(copy(X))) ≈ chebyshevutransform(chebyshevutransform(X,1),2)
            @test @inferred(chebyshevutransform(X,Val(2))) ≈ @inferred(chebyshevutransform!(copy(X),Val(2))) ≈ chebyshevutransform(chebyshevutransform(X,Val(2),1),Val(2),2)
        end

        @testset "ichebyshevutransform" begin
            @test @inferred(ichebyshevutransform(X,1)) ≈ @inferred(ichebyshevutransform!(copy(X),1)) ≈ hcat(ichebyshevutransform.([X[:,k] for k=axes(X,2)])...)
            @test ichebyshevutransform(X,2) ≈ ichebyshevutransform!(copy(X),2) ≈ hcat(ichebyshevutransform.([X[k,:] for k=axes(X,1)])...)'
            @test @inferred(ichebyshevutransform(X,Val(2),1)) ≈ @inferred(ichebyshevutransform!(copy(X),Val(2),1)) ≈ hcat(ichebyshevutransform.([X[:,k] for k=axes(X,2)],Val(2))...)
            @test ichebyshevutransform(X,Val(2),2) ≈ ichebyshevutransform!(copy(X),Val(2),2) ≈ hcat(ichebyshevutransform.([X[k,:] for k=axes(X,1)],Val(2))...)'

            @test @inferred(ichebyshevutransform(X)) ≈ @inferred(ichebyshevutransform!(copy(X))) ≈ ichebyshevutransform(ichebyshevutransform(X,1),2)
            @test @inferred(ichebyshevutransform(X,Val(2))) ≈ @inferred(ichebyshevutransform!(copy(X),Val(2))) ≈ ichebyshevutransform(ichebyshevutransform(X,Val(2),1),Val(2),2)

            @test ichebyshevutransform(chebyshevutransform(X)) ≈ X
            @test chebyshevutransform(ichebyshevutransform(X)) ≈ X
        end

        X = randn(1,1)
        @test chebyshevtransform!(copy(X), Val(1)) == ichebyshevtransform!(copy(X), Val(1)) == X
        @test_throws ArgumentError chebyshevtransform!(copy(X), Val(2))
        @test_throws ArgumentError ichebyshevtransform!(copy(X), Val(2))
    end

    @testset "tensor" begin
        @testset "3D" begin
            X = randn(4,5,6)
            X̃ = similar(X)
            @testset "chebyshevtransform" begin
                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = chebyshevtransform(X[:,k,j]) end
                @test @inferred(chebyshevtransform(X,1)) ≈ @inferred(chebyshevtransform!(copy(X),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = chebyshevtransform(X[k,:,j]) end
                @test chebyshevtransform(X,2) ≈ chebyshevtransform!(copy(X),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = chebyshevtransform(X[k,j,:]) end
                @test chebyshevtransform(X,3) ≈ chebyshevtransform!(copy(X),3) ≈ X̃

                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = chebyshevtransform(X[:,k,j],Val(2)) end
                @test @inferred(chebyshevtransform(X,Val(2),1)) ≈ @inferred(chebyshevtransform!(copy(X),Val(2),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = chebyshevtransform(X[k,:,j],Val(2)) end
                @test chebyshevtransform(X,Val(2),2) ≈ chebyshevtransform!(copy(X),Val(2),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = chebyshevtransform(X[k,j,:],Val(2)) end
                @test chebyshevtransform(X,Val(2),3) ≈ chebyshevtransform!(copy(X),Val(2),3) ≈ X̃

                @test @inferred(chebyshevtransform(X)) ≈ @inferred(chebyshevtransform!(copy(X))) ≈ chebyshevtransform(chebyshevtransform(chebyshevtransform(X,1),2),3)
                @test @inferred(chebyshevtransform(X,Val(2))) ≈ @inferred(chebyshevtransform!(copy(X),Val(2))) ≈ chebyshevtransform(chebyshevtransform(chebyshevtransform(X,Val(2),1),Val(2),2),Val(2),3)
            end

            @testset "ichebyshevtransform" begin
                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = ichebyshevtransform(X[:,k,j]) end
                @test @inferred(ichebyshevtransform(X,1)) ≈ @inferred(ichebyshevtransform!(copy(X),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = ichebyshevtransform(X[k,:,j]) end
                @test ichebyshevtransform(X,2) ≈ ichebyshevtransform!(copy(X),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = ichebyshevtransform(X[k,j,:]) end
                @test ichebyshevtransform(X,3) ≈ ichebyshevtransform!(copy(X),3) ≈ X̃

                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = ichebyshevtransform(X[:,k,j],Val(2)) end
                @test @inferred(ichebyshevtransform(X,Val(2),1)) ≈ @inferred(ichebyshevtransform!(copy(X),Val(2),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = ichebyshevtransform(X[k,:,j],Val(2)) end
                @test ichebyshevtransform(X,Val(2),2) ≈ ichebyshevtransform!(copy(X),Val(2),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = ichebyshevtransform(X[k,j,:],Val(2)) end
                @test ichebyshevtransform(X,Val(2),3) ≈ ichebyshevtransform!(copy(X),Val(2),3) ≈ X̃

                @test @inferred(ichebyshevtransform(X)) ≈ @inferred(ichebyshevtransform!(copy(X))) ≈ ichebyshevtransform(ichebyshevtransform(ichebyshevtransform(X,1),2),3)
                @test @inferred(ichebyshevtransform(X,Val(2))) ≈ @inferred(ichebyshevtransform!(copy(X),Val(2))) ≈ ichebyshevtransform(ichebyshevtransform(ichebyshevtransform(X,Val(2),1),Val(2),2),Val(2),3)

                @test ichebyshevtransform(chebyshevtransform(X)) ≈ X
                @test chebyshevtransform(ichebyshevtransform(X)) ≈ X
            end
        
            @testset "chebyshevutransform" begin
                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = chebyshevutransform(X[:,k,j]) end
                @test @inferred(chebyshevutransform(X,1)) ≈ @inferred(chebyshevutransform!(copy(X),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = chebyshevutransform(X[k,:,j]) end
                @test chebyshevutransform(X,2) ≈ chebyshevutransform!(copy(X),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = chebyshevutransform(X[k,j,:]) end
                @test chebyshevutransform(X,3) ≈ chebyshevutransform!(copy(X),3) ≈ X̃

                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = chebyshevutransform(X[:,k,j],Val(2)) end
                @test @inferred(chebyshevutransform(X,Val(2),1)) ≈ @inferred(chebyshevutransform!(copy(X),Val(2),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = chebyshevutransform(X[k,:,j],Val(2)) end
                @test chebyshevutransform(X,Val(2),2) ≈ chebyshevutransform!(copy(X),Val(2),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = chebyshevutransform(X[k,j,:],Val(2)) end
                @test chebyshevutransform(X,Val(2),3) ≈ chebyshevutransform!(copy(X),Val(2),3) ≈ X̃

                @test @inferred(chebyshevutransform(X)) ≈ @inferred(chebyshevutransform!(copy(X))) ≈ chebyshevutransform(chebyshevutransform(chebyshevutransform(X,1),2),3)
                @test @inferred(chebyshevutransform(X,Val(2))) ≈ @inferred(chebyshevutransform!(copy(X),Val(2))) ≈ chebyshevutransform(chebyshevutransform(chebyshevutransform(X,Val(2),1),Val(2),2),Val(2),3)
            end

            @testset "ichebyshevutransform" begin
                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = ichebyshevutransform(X[:,k,j]) end
                @test @inferred(ichebyshevutransform(X,1)) ≈ @inferred(ichebyshevutransform!(copy(X),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = ichebyshevutransform(X[k,:,j]) end
                @test ichebyshevutransform(X,2) ≈ ichebyshevutransform!(copy(X),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = ichebyshevutransform(X[k,j,:]) end
                @test ichebyshevutransform(X,3) ≈ ichebyshevutransform!(copy(X),3) ≈ X̃

                for k = axes(X,2), j = axes(X,3) X̃[:,k,j] = ichebyshevutransform(X[:,k,j],Val(2)) end
                @test @inferred(ichebyshevutransform(X,Val(2),1)) ≈ @inferred(ichebyshevutransform!(copy(X),Val(2),1)) ≈ X̃
                for k = axes(X,1), j = axes(X,3) X̃[k,:,j] = ichebyshevutransform(X[k,:,j],Val(2)) end
                @test ichebyshevutransform(X,Val(2),2) ≈ ichebyshevutransform!(copy(X),Val(2),2) ≈ X̃
                for k = axes(X,1), j = axes(X,2) X̃[k,j,:] = ichebyshevutransform(X[k,j,:],Val(2)) end
                @test ichebyshevutransform(X,Val(2),3) ≈ ichebyshevutransform!(copy(X),Val(2),3) ≈ X̃

                @test @inferred(ichebyshevutransform(X)) ≈ @inferred(ichebyshevutransform!(copy(X))) ≈ ichebyshevutransform(ichebyshevutransform(ichebyshevutransform(X,1),2),3)
                @test @inferred(ichebyshevutransform(X,Val(2))) ≈ @inferred(ichebyshevutransform!(copy(X),Val(2))) ≈ ichebyshevutransform(ichebyshevutransform(ichebyshevutransform(X,Val(2),1),Val(2),2),Val(2),3)

                @test ichebyshevutransform(chebyshevutransform(X)) ≈ X
                @test chebyshevutransform(ichebyshevutransform(X)) ≈ X
            end

            X = randn(1,1,1)
            @test chebyshevtransform!(copy(X), Val(1)) == ichebyshevtransform!(copy(X), Val(1)) == X
            @test_throws ArgumentError chebyshevtransform!(copy(X), Val(2))
            @test_throws ArgumentError ichebyshevtransform!(copy(X), Val(2))
        end

        @testset "4D" begin
            X = randn(2,3,4,5)
            X̃ = similar(X)
            for trans in (chebyshevtransform, ichebyshevtransform, chebyshevutransform, ichebyshevutransform)
                for k = axes(X,2), j = axes(X,3), l = axes(X,4) X̃[:,k,j,l] = trans(X[:,k,j,l]) end
                @test @inferred(trans(X,1)) ≈ X̃
                @test @inferred(trans(X)) ≈ trans(trans(trans(trans(X,1),2),3),4)
            end
        end
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
    @testset "BigInt" begin
        x = big(10)^400 .+ BigInt[1,2,3]
        @test ichebyshevtransform(chebyshevtransform(x)) ≈ x
    end

    @testset "immutable vectors" begin
        F = plan_chebyshevtransform([1.,2,3])
        @test chebyshevtransform(1.0:3) == F * (1:3)
        @test ichebyshevtransform(1.0:3) == ichebyshevtransform([1.0:3;])
    end

    @testset "inv" begin
        x = randn(5)
        for F in (plan_chebyshevtransform(x), plan_chebyshevtransform(x, Val(2)),
                  plan_chebyshevutransform(x), plan_chebyshevutransform(x, Val(2)),
                  plan_ichebyshevtransform(x), plan_ichebyshevtransform(x, Val(2)),
                  plan_ichebyshevutransform(x), plan_ichebyshevutransform(x, Val(2)))
            @test F \ (F*x) ≈ F * (F\x) ≈ x
        end

        X = randn(5,4)
        for F in (plan_chebyshevtransform(X,Val(1),1), plan_chebyshevtransform(X, Val(2),1),
            plan_chebyshevtransform(X,Val(1),2), plan_chebyshevtransform(X, Val(2),2),
            plan_ichebyshevtransform(X,Val(1),1), plan_ichebyshevtransform(X, Val(2),1),
            plan_ichebyshevtransform(X,Val(1),2), plan_ichebyshevtransform(X, Val(2),2))
            @test F \ (F*X) ≈ F * (F\X) ≈ X
        end
        # Matrix isn't implemented for chebyshevu
        for F in (plan_chebyshevutransform(X,Val(1),1), plan_chebyshevutransform(X, Val(2),1),
            plan_chebyshevutransform(X,Val(1),2), plan_chebyshevutransform(X, Val(2),2),
            plan_ichebyshevutransform(X,Val(1),1), plan_ichebyshevutransform(X, Val(2),1),
            plan_ichebyshevutransform(X,Val(1),2), plan_ichebyshevutransform(X, Val(2),2))
            @test F \ (F*X) ≈ F * (F\X) ≈ X
        end
    end

    @testset "incompatible shapes" begin
        @test_throws ErrorException plan_chebyshevtransform(randn(5)) * randn(5,5)
        @test_throws ErrorException plan_ichebyshevtransform(randn(5)) * randn(5,5)
    end

    @testset "plan via size" begin
        X = randn(3,4)
        p = plan_chebyshevtransform(Float64, (3,4))
        @test p * X == chebyshevtransform(X)
    end
end
