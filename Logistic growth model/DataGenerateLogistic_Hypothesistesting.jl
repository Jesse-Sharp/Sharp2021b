#=
    DataGenerateLogistic_HypothesisTesting.jl
    Produces synthetic data for the logistic model for hypothesis testing (random seed set in Logistic_Infer_r_C0_highcurvature_HT.jl)
=#

function DataGenerateLogistic(θ,T,NperT,σ)
    if length(NperT) == 1 #equal number of observations per time-point
        Data = zeros(NperT*length(T))
        Time = zeros(NperT*length(T))
        for (i,t) in enumerate(T)
            NormDist =  Normal(LogisticSolAllParams(θ,t),σ)
            Time[((i-1)*NperT+1):i*NperT] = t.*ones(NperT)
            #Data drawn from normal distribution with mean corresponding to the model behaviour
            Data[((i-1)*NperT+1):i*NperT] = rand(NormDist,NperT)
        end
    else
    #different number of observations per time-point
        Data = zeros(sum(NperT))
        Time = []
        for a in 1:length(NperT)
        append!(Time, T[a].*ones(NperT[a]))
        end
        for j in 1:sum(NperT)
            NormDist =  Normal(LogisticSolAllParams(θ,Time[j]),σ)
            #Data drawn from normal distribution with mean corresponding to the model behaviour
            Data[j] = rand(NormDist)
        end
    end
return Time, Data
end
