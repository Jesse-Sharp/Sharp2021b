#=
    FisherSIR.jl
    Fisher information for the SIR model where S, I and R are observed
=#

#With unknown parameters θ = (β,γ)
function Fisher_βγ(θ,σ,T,NperT,m)
    J = SIR_Jacobian_βγ(θ,T)
    FIM = J'*FIM_Obs_Process_SIR(σ,T,NperT,m)*J
    return FIM
end

#With unknown parameters θ = (β,σ)
function Fisher_βσ(θ,σ,T,NperT,m)
    J = SIR_Jacobian_βσ(θ,T)
    FIM = J'*FIM_Obs_Process_SIR(σ,T,NperT,m)*J
    return FIM
end
