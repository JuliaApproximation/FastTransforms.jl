#using FastTransforms
#using Base.Test


function nudft1{T<:AbstractFloat}( c::AbstractVector, x::AbstractVector{T} )
# Nonuniform discrete Fourier transform of type 1 

N = size(x, 1)
output = zeros(c)
er = collect(0:N-1)
for j = 1 : N 
	output[j] = dot( exp(2*pi*1im*x[j]*er), c) 
end 

return output
end

function nudft2{T<:AbstractFloat}( c::AbstractVector, ω::AbstractVector{T}) 
# Nonuniform discrete Fourier transform of type II

N = size(ω, 1)
output = zeros(c)
for j = 1 : N
	output[j] = dot( exp(2*pi*1im*(j-1)/N*ω), c)
end

return output
end


# Test nufft1(): 
ϵ = eps(Float64)
for n = 10.^(0:4)
	c = complex(rand(n))
	x = pop!(collect(linspace(0,1,n+1)));
	x += 3*rand(n)/n
	exact = nudft1( c, x )
	fast = nufft1( c, x, ϵ )
	@test norm(exact - fast,Inf) < 500*ϵ*n*norm(c)
end  

# Test nufft2(): 
ϵ = eps(Float64)
for n = 10.^(0:4)
	c = complex(rand(n))
	ω = collect(0:n-1)
	ω += rand(n)
	exact = nudft2( c, ω )
	fast = nufft2( c, ω, ϵ )
	@test norm(exact - fast,Inf) < 500*ϵ*n*norm(c)
end  


