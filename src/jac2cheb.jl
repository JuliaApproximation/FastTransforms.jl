function jac2cheb{T<:AbstractFloat}(c_jac::AbstractVector{T},α::T,β::T,plan::ChebyshevJacobiPlan{FORWARD,T})
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
