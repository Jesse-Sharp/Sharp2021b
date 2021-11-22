#=
    SIR_Jacobian.jl
    Forms the Jacobian of the SIR model with respect to the parameters,  with observations on S, I and R;  as required for computing the Fisher information matrix
=#

#With unknown parameters θ = (β,γ)
function SIR_Jacobian_βγ(θ,T)
        J = zeros(2*length(T)*3,2)
        for (i,t) in enumerate(T)
            (dSdβ, dIdβ, dRdβ) = dModeldParams(θ,t)[:,1]
            (dSdγ, dIdγ, dRdγ) = dModeldParams(θ,t)[:,2]
        #Here we skip every second row as the parameters we consider are means so d/dσ is zero
        J[6*i-5,:] = [dSdβ dSdγ] #dModelOutput/dparam
        J[6*i-3,:] = [dIdβ dIdγ] #dModelOutput/dparam
        J[6*i-1,:] = [dRdβ dRdγ] #dModelOutput/dparam
        end
    return J
end

#With unknown parameters θ = (β,σ)
function SIR_Jacobian_βσ(θ,T)
    #Currently this assumes all 3 variables are observed (SIR)
        J = zeros(2*length(T)*3,2)
        for (i,t) in enumerate(T)
            (dSdβ, dIdβ, dRdβ) = dModeldParams(θ,t)[:,1]
        #Every second row is d/dσ (mean,σ)
        J[6*i-5,:] = [dSdβ 0] #dModelOutput/dparam
        J[6*i-4,:] = [0 1]
        J[6*i-3,:] = [dIdβ 0] #dModelOutput/dparam
        J[6*i-2,:] = [0 1]
        J[6*i-1,:] = [dRdβ 0] #dModelOutput/dparam
        J[6*i,:] = [0 1]
        end
    return J
end
