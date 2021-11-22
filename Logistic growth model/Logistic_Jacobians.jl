#=
    Logistic_Jacobians.jl
    Forms the Jacobian of the logistic model with respect to the parameters;  as required for computing the Fisher information matrix
=#

#With unknown parameters θ = (r,C(0))
function Logistic_Jacobian_rC(θ,T)
    J = zeros(2*length(T),2)
    for (i,t) in enumerate(T)
    #Here we skip every second row as the parameters we consider are means so d/dσ is zero
    J[2*i-1,:] = [dLdr(θ,t) dLdC0(θ,t)] #dModelOutput/dparam
    end
    return J
end

#With unknown parameters θ = (r,K)
function Logistic_Jacobian_rK(θ,T)
    J = zeros(2*length(T),2)
    for (i,t) in enumerate(T)
    #Here we skip every second row as the parameters we consider are means so d/dσ is zero
    J[2*i-1,:] = [dLdr(θ,t) dLdK(θ,t)] #dModelOutput/dparam
    end
    return J
end
