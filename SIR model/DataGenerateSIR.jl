#=
    DataGenerateSIR.jl
    Produces synthetic data for the SIR model where S, I and R are observed
=#

function DataGenerateSIR(θ,T,NperT,σ)
    Random.seed!(4107)
    sol = SIRSolAllParams(θ,maximum(T))
    if length(NperT) == 1 #equal number of observations per time-point
        Data = zeros(NperT*length(T),3)
        Time = zeros(NperT*length(T))
        for (i,t) in enumerate(T)
            NormDistS =  Normal.(sol(t)[1],σ)
            NormDistI =  Normal.(sol(t)[2],σ)
            NormDistR =  Normal.(sol(t)[3],σ)
            Time[((i-1)*NperT+1):i*NperT] = t.*ones(NperT)
            #Data drawn from normal distribution with mean corresponding to the model behaviour
            Data[((i-1)*NperT+1):i*NperT,1] = rand(NormDistS,NperT)
            Data[((i-1)*NperT+1):i*NperT,2] = rand(NormDistI,NperT)
            Data[((i-1)*NperT+1):i*NperT,3] = rand(NormDistR,NperT)
        end
    else
    #different number of observations per time-point
        Data = zeros(sum(NperT),3)
        Time = []
        for a in 1:length(NperT)
        append!(Time, T[a].*ones(NperT[a]))
        end
        for j in 1:sum(NperT)
            NormDistS =  Normal.(sol(Time[j])[1],σ)
            NormDistI =  Normal.(sol(Time[j])[2],σ)
            NormDistR =  Normal.(sol(Time[j])[3],σ)
            Data[j,1] = rand(NormDistS)
            Data[j,2] = rand(NormDistI)
            Data[j,3] = rand(NormDistR)
        end
    end
return Time, Data
end
