#=
    DataGenerateSIR_obsIonly.jl
    Produces synthetic data for the SIR model where only I is observed
=#

function DataGenerateSIR_obsIonly(θ,T,NperT,σ)
    Random.seed!(4107)
    sol = SIRSolAllParams(θ,maximum(T))

    if length(NperT) == 1 #equal number of observations per time-point
        Data = zeros(NperT*length(T),1)
        Time = zeros(NperT*length(T))
        for (i,t) in enumerate(T)
            NormDistI =  Normal.(sol(t)[2],σ)
            Time[((i-1)*NperT+1):i*NperT] = t.*ones(NperT)
            #Data drawn from normal distribution with mean corresponding to the model behaviour
            Data[((i-1)*NperT+1):i*NperT,1] = rand(NormDistI,NperT)
        end
    else
    #different number of observations per time-point
        Data = zeros(sum(NperT),1)
        Time = []
        for a in 1:length(NperT)
        append!(Time, T[a].*ones(NperT[a]))
        end
        for j in 1:sum(NperT)
            NormDistI =  Normal.(sol(Time[j])[2],σ)
            #Data drawn from normal distribution with mean corresponding to the model behaviour
            Data[j,1] = rand(NormDistI)
        end
    end
return Time, Data
end
