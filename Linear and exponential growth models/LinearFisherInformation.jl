#=
    LinearFisherInformation.jl
    Fisher information for the linear model
=#

#With unknown parameters θ = (a,C(0))
function Fisher_aC0(θ,σ,T,NperT)
    J = Linear_Jacobian_aC0(θ,T)
    FIM = J'*FIM_Obs_Process(σ,T,NperT)*J
    return FIM
end

#With unknown parameters θ = (a,σ)
function Fisher_a_StDev(θ,σ,T,NperT)
    J = Linear_Jacobian_a_StDev(θ,T)
    FIM = J'*FIM_Obs_Process(σ,T,NperT)*J
    return FIM
end

#With unknown parameters θ = (C(0),σ)
function Fisher_C0_StDev(θ,σ,T,NperT)
    J = Linear_Jacobian_C0_StDev(θ,T)
    FIM = J'*FIM_Obs_Process(σ,T,NperT)*J
    return FIM
end
