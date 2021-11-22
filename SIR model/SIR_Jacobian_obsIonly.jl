#=
    SIR_Jacobian_obsIonly.jl
    Forms the Jacobian of the SIR model with respect to the parameters,  with observations on I only;  as required for computing the Fisher information matrix.
    We compute the partial derivatives of SIR model with respect to the parameters using forward mode automatic differentiation
=#

#With unknown parameters θ = (β,γ)
function SIR_Jacobian_βγ_obsIonly(θ,T)
        J = zeros(2*length(T),2)
        for (i,t) in enumerate(T)
            SIRsol = ϕ -> SIRSolAllParams([ϕ;θ[3:5]],t).u[end,:][1]
            J[2*i-1,:] = ForwardDiff.jacobian(SIRsol,θ[1:2])[2,:]
        end
    return J
end

#With unknown parameters θ = (β,σ)
function SIR_Jacobian_βσ_obsIonly(θ,T)
    J = zeros(2*length(T),2)
    for (i,t) in enumerate(T)
            SIRsolβ = ϕ -> SIRSolAllParams([ϕ;θ[2:5]],t).u[end,:][1]
            dIdβ = ForwardDiff.derivative(SIRsolβ,θ[1])[2]
    J[2*i-1,:] = [dIdβ 0]
    J[2*i,:] = [0 1]
    end
    return J
end
