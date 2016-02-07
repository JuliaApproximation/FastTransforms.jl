jac2cheb{T<:AbstractFloat}(c::Vector{T},α::Number,β::Number) = jac2cheb(c,convert(T,α),convert(T,β))

jac2cheb{T<:AbstractFloat}(c::Vector{T},α::T,β::T) = jac2cheb(c,α,β,ForwardChebyshevJacobiPlan(c,α,β))

function jac2cheb{T<:AbstractFloat}(c_jac::Vector{T},α::T,β::T,plan::ChebyshevJacobiPlan{FORWARD,T})
    #if α == β return jac2cheb(c_jac,α) end

    M,N,nM₀,αN,K = getconstants(plan)
    i₁,i₂,j₁,j₂ = getindices(plan)
    p₁,p₂,rp,c₁,c₂,um,vm,cfs,θ,tempcos,tempsin,tempcosβsinα,tempmindices,cnαβ,cnmαβ = getplan(plan)

    v_cheb = zero(c_jac)

    k=0
    for k=1:K
        copy!(cnmαβ,cnαβ)

        j₁[k] < nM₀ && (k-=1; break)

        @inbounds for i=i₁[k]:i₁[k+1]-1 v_cheb[i] += backward_recurrence(c_jac,j₂[k],θ[i],tempcos[i],tempsin[i],rp) end
        @inbounds for i=i₂[k+1]+1:i₂[k] v_cheb[i] += backward_recurrence(c_jac,j₂[k],θ[i],tempcos[i],tempsin[i],rp) end

        for m=0:M-1
            # Initialize diagonal N-scaling multiplied by local coefficients and zero out excess
            init_c₁c₂!(c₁,c₂,cnmαβ,c_jac,j₁[k],j₂[k])

            # Apply planned DCT-I and DST-I in-place
            applyTN!(c₁,p₁),applyUN!(c₂,p₂) # 1 allocation from slicing the Array

            # Compute u_m(θ) and v_m(θ)
            compute_umvm!(um,vm,cfs,α,β,tempcos,tempsin,tempcosβsinα,m,θ,i₁[k+1]:i₂[k+1])

            # Multiply point-wise by u_m(θ) and v_m(θ) for valid indices
            @inbounds for i=i₁[k+1]:i₂[k+1] v_cheb[i] += um[i]*c₁[i] + vm[i]*c₂[i] end

            # Update C_{n,m}^{α,β} by recurrence in m
            @inbounds for i=1:N+1 cnmαβ[i] /= 2(2i+α+β+m) end
        end
    end

    # Finish off recurrence
    @inbounds for i=i₁[k+1]:i₂[k+1] v_cheb[i] += backward_recurrence(c_jac,j₂[k+1],θ[i],tempcos[i],tempsin[i],rp) end

    # perform IDCT-I
    applyTNinv!(v_cheb,p₁)
end


# If α == β == λ - 1/2
#=
function jac2cheb{T<:AbstractFloat}(c_jac::Vector{T},λ::T)
    M,N,λ = 7,length(c_jac)-1,λ+one(T)/2

    θ = T[k/N for k=zero(T):N] # Clenshaw-Curtis points

    nM₀ = min(floor(Int,(eps(T)*sqrtpi*2^(M-λ-1)*factorial(M)/pochhammer(λ,M)/pochhammer(1-λ,M))^(-1/(M+1/2))),N)
    αN = min(1/log(N/nM₀),1/2)
    K = ceil(Int,log(N/nM₀)/log(1/αN))

    # Initialize DCT-I and DST-I plans
    p₁,p₂ = applyTN_plan(c_jac),applyUN_plan(c_jac)

    rp = RecurrencePlan(λ-1/2,λ-1/2,N+1)

    # Initialize local coefficients and values at Clenshaw-Curtis points
    c₁,c₂,v_cheb = zero(c_jac),zero(c_jac),zero(c_jac)

    # Initialize array to store coefficients
    um,vm = zero(c_jac),zero(c_jac)
    tempsin,tempsinλ,tempmindices = sinpi(θ),zero(c_jac),zero(c_jac)
    @inbounds for i=1:N+1 tempsinλ[i] = tempsin[i]^λ end

    cnλ = Cnλ(0:N,λ)
    cnmλ = similar(cnλ)

    i₁old,i₂old,jₖ₂ = 1,N+1,N+1

    for k=1:K
        copy!(cnmλ,cnλ)

        jₖ₁ = ceil(Int,αN^k*N)
        if jₖ₁ < nM₀ break end
        i₁,i₂ = findmindices!(tempmindices,λ,jₖ₁,M,tempsin,tempsinλ)
        #println("This is i₁:i₂: ",i₁,":",i₂," and this is j₁:j₂: ",jₖ₁,":",jₖ₂)

        @inbounds for i=i₁old:i₁-1 v_cheb[i] += backward_recurrence(c_jac,jₖ₂,θ[i],rp) end
        @inbounds for i=i₂+1:i₂old v_cheb[i] += backward_recurrence(c_jac,jₖ₂,θ[i],rp) end

        for m=0:M-1
            # Initialize diagonal N-scaling multiplied by local coefficients and zero out excess
            init_c₁c₂!(c₁,c₂,cnmλ,c_jac,jₖ₁,jₖ₂)

            # Apply planned DCT-I and DST-I in-place
            applyTN!(c₁,p₁),applyUN!(c₂,p₂) # 1 allocation from slicing the Array

            # Compute u_m(θ) and v_m(θ)
            compute_umvm!(um,vm,λ,tempsin,tempsinλ,m,θ,i₁:i₂)

            # Muliply point-wise by u_m(θ) and v_m(θ) for valid indices

            @inbounds for i=i₁:i₂ v_cheb[i] += um[i]*c₁[i] + vm[i]*c₂[i] end

            # Updated C_{n,m}^λ by recurrence in m
            @inbounds for i=1:N+1 cnmλ[i] *= (λ+m)/2/(m+1)*(m+1-λ)/(i+λ+m) end
        end
        i₁old,i₂old,jₖ₂ = i₁,i₂,jₖ₁-1
    end

    # Finish off recurrence
    @inbounds for i=i₁old:i₂old v_cheb[i] += backward_recurrence(c_jac,jₖ₂,θ[i],rp) end

    # perform IDCT-I
    applyTNinv!(v_cheb,p₁)
end
=#
