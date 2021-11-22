#=
    LogisticFisherInformation.jl
    Fisher information for the logistic model
=#

#With unknown parameters θ = (r,C(0))
function Fisher_rC(θ,σ,T,NperT)
    J = Logistic_Jacobian_rC(θ,T)
    FIM = J'*FIM_Obs_Process(σ,T,NperT)*J
    return FIM
end

#With unknown parameters θ = (r,K)
function Fisher_rK(θ,σ,T,NperT)
    J = Logistic_Jacobian_rK(θ,T)
    FIM = J'*FIM_Obs_Process(σ,T,NperT)*J
    return FIM
end
