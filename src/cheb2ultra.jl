function cheb2ultra(c_cheb::AbstractVector{T},λ::T,plan::ChebyshevUltrasphericalPlan{BACKWARD,T}) where T<:AbstractFloat
    M,N,nM₀,αN,K = getconstants(plan)
    i₁,i₂,j₁,j₂ = getindices(plan)
    p₁,p₂,rp,c₁,c₂,um,vm,θ,tempsin,tempsin2,tempsinλ,tempsinλm,tempmindices,cnλ,cnmλ,w,anλ,c_cheb2,pr = getplan(plan)

    # Initialize c_cheb2
    @inbounds for i=1:N+1 c_cheb2[i] = c_cheb[i] end
    @inbounds for i=N+2:2N+1 c_cheb2[i] = 0 end

    # Initialize values at Clenshaw-Curtis points
    v_ultra = zero(c_cheb2)

    # Perform transposed DCT-I
    applyTN!(c_cheb2,p₁)

    # Scale values by Clenshaw-Curtis weights
    @inbounds for i=1:2N+1 c_cheb2[i] *= w[i] end

    k=0
    for κ=1:K
        k = κ
        copyto!(cnmλ,cnλ)
        copyto!(tempsinλm,tempsinλ)

        j₁[k] < nM₀ && (k-=1; break)

        @inbounds for i=i₁[k]:i₁[k+1]-1
            forward_recurrence!(pr,j₂[k],θ[i],tempsin2[2N+2-i],tempsin2[i],rp)
            cc2i = c_cheb2[i]
            for j=1:j₂[k] v_ultra[j] += cc2i*pr[j] end
        end

        @inbounds for i=i₂[k+1]+1:i₂[k]
            forward_recurrence!(pr,j₂[k],θ[i],tempsin2[2N+2-i],tempsin2[i],rp)
            cc2i = c_cheb2[i]
            for j=1:j₂[k] v_ultra[j] += cc2i*pr[j] end
        end

        for m=0:M-1
            # Compute u_m(θ) and v_m(θ)
            compute_umvm!(um,vm,λ,tempsinλm,m,θ,i₁[k+1]:i₂[k+1])

            # Multiply point-wise by u_m(θ) and v_m(θ) for valid indices
            init_c₁c₂!(c₁,c₂.parent,um,vm,c_cheb2,i₁[k+1],i₂[k+1])

            # Apply planned DCT-I and DST-I in-place
            applyTN!(c₁,p₁);applyUN!(c₂,p₂)

            # Compute diagonal 2N-scaling multiplied by local coefficients and zero out excess
            @inbounds for j=j₁[k]:j₂[k] v_ultra[j] += cnmλ[j]*(c₁[j]+c₂.parent[j]) end

            # Update C_{n,m}^λ by recurrence in m
            cst = (λ+m)/2/(m+1)*(m+1-λ)
            @inbounds for j=1:2N+1 cnmλ[j] *= cst/(j+λ+m) end

            # Update sin^{λ+m}(θ)
            @inbounds for j=1:2N+1 tempsinλm[j] *= tempsin[j] end
        end
    end

    # Finish off recurrence
    @inbounds for i=i₁[k+1]:i₂[k+1]
        forward_recurrence!(pr,j₂[k+1],θ[i],tempsin2[2N+2-i],tempsin2[i],rp)
        cc2i = c_cheb2[i]
        for j=1:j₂[k+1] v_ultra[j] += cc2i*pr[j] end
    end

    # Scale by orthonormality constants
    ret = zero(c_cheb)
    @inbounds for i=1:N+1 ret[i] = v_ultra[i]/anλ[i] end

    ret
end
