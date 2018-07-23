function cheb2jac(c_cheb::AbstractVector{T},α::T,β::T,plan::ChebyshevJacobiPlan{BACKWARD,T}) where T<:AbstractFloat
    M,N,nM₀,αN,K = getconstants(plan)
    i₁,i₂,j₁,j₂ = getindices(plan)
    p₁,p₂,rp,c₁,c₂,um,vm,cfs,θ,tempcos,tempsin,tempcosβsinα,tempmindices,cnαβ,cnmαβ,w,anαβ,c_cheb2,pr = getplan(plan)

    # Initialize c_cheb2
    @inbounds for i=1:N+1 c_cheb2[i] = c_cheb[i] end
    @inbounds for i=N+2:2N+1 c_cheb2[i] = 0 end

    # Initialize values at Clenshaw-Curtis points
    v_jac = zero(c_cheb2)

    # Perform transposed DCT-I
    p₁*c_cheb2

    # Scale values by Clenshaw-Curtis weights
    @inbounds for i=1:2N+1 c_cheb2[i] *= w[i] end

    k=0
    for κ = 1:K
        k = κ
        copyto!(cnmαβ,cnαβ)

        j₁[k] < nM₀ && (k-=1; break)

        @inbounds for i=i₁[k]:i₁[k+1]-1
            forward_recurrence!(pr,j₂[k],θ[i],tempcos[i],tempsin[i],rp)
            cc2i = c_cheb2[i]
            for j=1:j₂[k] v_jac[j] += cc2i*pr[j] end
        end

        @inbounds for i=i₂[k+1]+1:i₂[k]
            forward_recurrence!(pr,j₂[k],θ[i],tempcos[i],tempsin[i],rp)
            cc2i = c_cheb2[i]
            for j=1:j₂[k] v_jac[j] += cc2i*pr[j] end
        end

        for m=0:M-1
            # Compute u_m(θ) and v_m(θ)
            compute_umvm!(um,vm,cfs,α,β,tempcos,tempsin,tempcosβsinα,m,θ,i₁[k+1]:i₂[k+1])

            # Multiply point-wise by u_m(θ) and v_m(θ) for valid indices
            init_c₁c₂!(c₁,c₂.parent,um,vm,c_cheb2,i₁[k+1],i₂[k+1])

            # Apply planned DCT-I and DST-I in-place
            p₁*c₁; p₂*c₂

            # Compute diagonal 2N-scaling multiplied by local coefficients and zero out excess
            @inbounds for j=j₁[k]:j₂[k] v_jac[j] += cnmαβ[j]*(c₁[j]+c₂.parent[j]) end

            # Update C_{n,m}^{α,β} by recurrence in m
            @inbounds for j=1:2N+1 cnmαβ[j] /= 2(2j+α+β+m) end
        end
    end

    # Finish off recurrence
    @inbounds for i=i₁[k+1]:i₂[k+1]
        forward_recurrence!(pr,j₂[k+1],θ[i],tempcos[i],tempsin[i],rp)
        cc2i = c_cheb2[i]
        for j=1:j₂[k+1] v_jac[j] += cc2i*pr[j] end
    end

    # Scale by orthonormality constants
    ret = zero(c_cheb)
    @inbounds for i=1:N+1 ret[i] = v_jac[i]/anαβ[i] end

    ret
end
