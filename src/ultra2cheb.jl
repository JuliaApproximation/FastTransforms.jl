function ultra2cheb(c_ultra::AbstractVector{T},λ::T,plan::ChebyshevUltrasphericalPlan{FORWARD,T}) where T<:AbstractFloat
    M,N,nM₀,αN,K = getconstants(plan)
    i₁,i₂,j₁,j₂ = getindices(plan)
    p₁,p₂,rp,c₁,c₂,um,vm,θ,tempsin,tempsin2,tempsinλ,tempsinλm,tempmindices,cnλ,cnmλ = getplan(plan)

    v_cheb = zero(c_ultra)

    k=0
    for k=1:K
        copy!(cnmλ,cnλ)
        copy!(tempsinλm,tempsinλ)

        j₁[k] < nM₀ && (k-=1; break)

        @inbounds for i=i₁[k]:i₁[k+1]-1 v_cheb[i] += backward_recurrence(c_ultra,j₂[k],θ[i],tempsin2[N+2-i],tempsin2[i],rp) end
        @inbounds for i=i₂[k+1]+1:i₂[k] v_cheb[i] += backward_recurrence(c_ultra,j₂[k],θ[i],tempsin2[N+2-i],tempsin2[i],rp) end

        for m=0:M-1
            # Initialize diagonal N-scaling multiplied by local coefficients and zero out excess
            init_c₁c₂!(c₁,c₂.parent,cnmλ,c_ultra,j₁[k],j₂[k])

            # Apply planned DCT-I and DST-I in-place
            applyTN!(c₁,p₁);applyUN!(c₂,p₂)

            # Compute u_m(θ) and v_m(θ)
            compute_umvm!(um,vm,λ,tempsinλm,m,θ,i₁[k+1]:i₂[k+1])

            # Multiply point-wise by u_m(θ) and v_m(θ) for valid indices
            @inbounds for i=i₁[k+1]:i₂[k+1] v_cheb[i] += um[i]*c₁[i] + vm[i]*c₂.parent[i] end

            # Update C_{n,m}^λ by recurrence in m
            cst = (λ+m)/2/(m+1)*(m+1-λ)
            @inbounds for i=1:N+1 cnmλ[i] *= cst/(i+λ+m) end

            # Update sin^{λ+m}(θ)
            @inbounds for i=1:N+1 tempsinλm[i] *= tempsin[i] end
        end
    end

    # Finish off recurrence
    @inbounds for i=i₁[k+1]:i₂[k+1] v_cheb[i] += backward_recurrence(c_ultra,j₂[k+1],θ[i],tempsin2[N+2-i],tempsin2[i],rp) end

    # perform IDCT-I
    applyTNinv!(v_cheb,p₁)
end
