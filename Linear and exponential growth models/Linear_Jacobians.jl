#=
    Linear_Jacobians.jl
    Forms the Jacobian of the linear model with respect to the parameters;  as required for computing the Fisher information matrix
=#

#With unknown parameters θ = (a,C(0))
function Linear_Jacobian_aC0(θ,T)
    J = zeros(2*length(T),2)
    for (i,t) in enumerate(T)
    #Here we skip every second row as the parameters we consider are means so d/dσ is zero
    J[2*i-1,:] = [dCda(θ,t) dCdC0(θ,t)] #dModelOutput/dparam
    end
    return J
end

#With unknown parameters θ = (a,σ)
function Linear_Jacobian_a_StDev(θ,T)
    J = zeros(2*length(T),2)
    for (i,t) in enumerate(T)
    #Every second row is d/dσ (mean,σ)
    J[2*i-1,:] = [dCda(θ,t) 0] #dModelOutput/dparam
    J[2*i,:] = [0 1]
    end
    return J
end

#With unknown parameters θ = (C(0),σ)
function Linear_Jacobian_C0_StDev(θ,T)
    J = zeros(2*length(T),2)
    for (i,t) in enumerate(T)
    #Every second row is d/dσ (mean,σ)
    J[2*i-1,:] = [dCdC0(θ,t) 0] #dModelOutput/dparam
    J[2*i,:] = [0 1]
    end
    return J
end
