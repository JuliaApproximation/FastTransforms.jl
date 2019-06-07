using FastTransforms, Compat
using Compat.Test

import FastTransforms: δ

@testset "Gaunt coefficients" begin
    println("Testing Table 2 of Y.-l. Xu, JCAM 85:53–65, 1997.")
    for (m,n) in ((0,2),(1,2),(1,8),(6,8),(3,18),
                  (10,18),(5,25),(-23,25),(2,40),(-35,40),
                  (28,62),(-42,62),(1,99),(90,99),(10,120),
                  (80,120),(23,150),(88,150))
        @test norm(gaunt(m,n,-m,n)[end]./(big(-1.0)^m/(2n+1))-1, Inf) < 400eps()
    end
    println("Testing Table 3 of Y.-l. Xu, JCAM 85:53–65, 1997.")
    for (m,n,μ,ν) in ((0,1,0,5),(0,5,0,10),(0,9,0,10),(0,10,0,12),
                      (0,11,0,15),(0,12,0,20),(0,20,0,45),(0,40,0,80),
                      (0,45,0,100),(3,5,-3,6),(4,9,-4,15),(-8,18,8,23),
                      (-10,20,10,30),(5,25,-5,45),(15,50,-15,60),(-28,68,28,75),
                      (32,78,-32,88),(45,82,-45,100))
        @test norm(sum(gaunt(m,n,μ,ν))-δ(m,0), Inf) < 15000eps()
    end
end
