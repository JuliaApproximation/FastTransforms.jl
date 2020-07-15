using FastTransforms, Test
import FastTransforms: clenshaw, clenshaw!

@testset "clenshaw" begin
    @testset "Chebyshev" begin
        @test @inferred(clenshaw([1,2,3],1)) ≡ 1 + 2 + 3
        @test @inferred(clenshaw([1,2,3],0)) ≡ 1 + 0 - 3
        @test @inferred(clenshaw([1,2,3],0.1)) == 1 + 2*0.1 + 3*cos(2acos(0.1))
        @test @inferred(clenshaw([1,2,3],[-1,0,1])) == clenshaw!([1,2,3],[-1,0,1]) == [2,-2,6]
        @test clenshaw([1,2,3],[-1,0,1]) isa Vector{Int}

        x = [1,0,0.1]
        @test @inferred(clenshaw([1,2,3],x)) == @inferred(clenshaw!([1,2,3],copy(x))) == 
            @inferred(clenshaw!([1,2,3],x,similar(x))) ==
            @inferred(clenshaw([1.,2,3],x)) == @inferred(clenshaw!([1.,2,3],copy(x))) == 
            @inferred(clenshaw!([1.,2,3],x,similar(x))) == [6,-2,-1.74]
    end
end