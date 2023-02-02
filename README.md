### Sharp2021b ###

Julia implementation of likelihood based inference and information geometry techniques, including geodesic curves and scalar curvature. This code is used to generate results by Sharp et al. in:

Sharp JA, Browning AP, Burrage K, Simpson MJ. 2022 Parameter estimation and uncertainty quantification using information geometry. _Journal of the Royal Society Interface_. (https://royalsocietypublishing.org/doi/full/10.1098/rsif.2021.0940).

A rough estimate of the order of the run-time of user callable functions is provided as a guide. These estimates are obtained on a standard university issued desktop computer (Dell Optiplex 7050), assuming that the required Julia packages are already precompiled.  
## This repository contains the following code and folders: ##

Contains code for reproducing the inference and information geometry results for the normal distributions, presented in Figures 1 and 2 of the Sharp et al. paper. Also includes supporting functions for producing optimisation, producing confidence regions, and generating information geometry results, that are used by all user callable functions in this repository. 

**The user callable scripts include:**

_NormalModel.jl_

- Code for reproducing the inference and information geometry results for the univariate normal distribution.  

_MvNormalModel.jl_

- Code for reproducing the inference and information geometry results for the multivariate normal distribution. 

_NormalModel_HT.jl_

- Code for reproducing the individual hypothesis test results for the univariate normal distribution.  

_MvNormalModel_HT.jl_

- Code for reproducing the individual hypothesis test results for the multivariate normal distribution. 

_NormalModel_HypothesisTests_Loop.jl_

- Code for reproducing the ensemble hypothesis test results for the univariate normal distribution.  

_MvNormalModel_HypothesisTests_Loop.jl_

- Code for reproducing the ensemble hypothesis test results for the multivariate normal distribution. 

The runtime for these scripts is less than one minute.  

**The supporting functions and data are:**

_ConfidenceRegions.jl_

- Contains the _confidenceregion2D_ function, for producing confidence regions in two dimensions. 

_EquidistantCircumferencePoints.jl_

- Contains the _EquidistantCircumferencePoints_ function for producing initial velocities for geodesics, corresponding to equally distant points on the circumference of a unit circle. 

_InfoGeo2D.jl_

- Contains functions for computing information geometry quantities in two dimensions; including Christoffel symbols of the first and second kind (_Christoffel2D_, _Christoffel2D\_first_), the right hand side of the geodesic ODEs (_ODEproblem\_geodesic2D!_), functions for solving geodesic ODEs to a specified geodesic length (_SingleGeodesic2D_, _conditionstop\_shooting1_, _conditionstop\_shooting2_, _conditionstop\_shooting3_, _affectstop!_), and a function for computing the scalar curvature, via the Riemann tensor and Ricci tensor (_RiemannTensor_).    

_optimise.jl_

- Contains the _optimise_ function for performing nonlinear bound constrained optimisation using the BOBYQA algorithm

#### Linear and exponential growth models ####

Contains code for reproducing the inference and information geometry results for the linear and exponential growth models, presented in Figures 4 and 6 of the Sharp et al. paper. 

**The user callable scripts include:**

_Linear\_Infer\_a\_C0.jl_

- Reproduces inference and information geometry results for the linear model, estimating unknown parameters _a_ and _C_(0).

_Exponential\_Infer\_a\_C0.jl_

- Reproduces inference and information geometry results for the exponential model, estimating unknown parameters _a_ and _C_(0).

_Linear\_Infer\_a\_StDev.jl_

- Reproduces inference and information geometry results for the linear model, estimating unknown parameters _a_ and the standard deviation of the data.

_Exponential\_Infer\_a\_StDev.jl_

- Reproduces inference and information geometry results for the exponential model, estimating unknown parameters _a_ and the standard deviation of the data.

_Linear\_Infer\_a\_StDev.jl_

- Reproduces inference and information geometry results for the linear model, estimating unknown parameter _C_(0) and the standard deviation of the data.

_Exponential\_Infer\_a\_StDev.jl_

- Reproduces inference and information geometry results for the exponential model, estimating unknown parameter _C_(0) and the standard deviation of the data.

The runtime of each of these scripts is on the order of one minute.

**The supporting functions and data are:**

_DataGenerateLinear.jl_

- Generates synthetic data for the linear model.

_DataGenerateExponential.jl_

- Generates synthetic data for the exponential model.

_FIM\_Obs\_Process.jl_

- Computes the Fisher information matrix for the observation process of the linear and exponential models.

_LinearFisherInformation.jl_

- Computes the Fisher information matrix for the linear model.

_ExponentialFisherInformation.jl_

- Computes the Fisher information matrix for the exponential model.

_Linear\_Jacobian.jl_

- Computes the Jacobian of the linear model with respect to the parameters.

_Exponential\_Jacobian.jl_

- Computes the Jacobian of the exponential model with respect to the parameters.

_Linear\_partials.jl_

- Computes the partial derivatives of the linear model with respect to the parameters, required to form Jacobian matrices.

_Exponential\_partials.jl_

- Computes the partial derivatives of the exponential model with respect to the parameters, required to form Jacobian matrices.

_LinearSolAllParams.jl_

- Computes the analytical solution of the Linear model. 

_ExponentialSolAllParams.jl_

- Computes the analytical solution of the exponential model. 


#### Logistic growth model ####

Contains code for reproducing the inference and information geometry results for the logistic growth model, presented in Figures 3, 5, 7 and 8 of the Sharp et al. paper. 

**The user callable scripts include:**

_Logistic\_Infer\_r\_C0.jl_

- Reproduces inference and information geometry results for the logistic model, estimating unknown parameters _r_ and _C_(0).

_Logistic\_Infer\_r\_C0_highcurvature.jl_

- Reproduces inference and information geometry results for the logistic model, estimating unknown parameters _r_ and _C_(0) in the high curvature region.

_Logistic\_Infer\_r\_K.jl_

- Reproduces inference and information geometry results for the logistic model, estimating unknown parameters _r_ and _K_.

_Logistic\_Infer\_r\_C0_highcurvature_HT.jl_

- Reproduces hypothesis test results for the logistic model, estimating unknown parameters _r_ and _C_(0) in the high curvature region.

The runtime of each of these scripts is on the order of one minute.

**The supporting functions and data are:**

_DataGenerateLogistic.jl_

- Generates synthetic data for the logistic model.

_DataGenerateLogistic_Hypothesistesting.jl_

- Generates synthetic data for the logistic model, but does not set the random seed; so that it can be called in a loop to generate synthetic datasets.

_FIM\_Obs\_Process.jl_

- Computes the Fisher information matrix for the observation process of the logistic model.

_LogisticFisherInformation.jl_

- Computes the Fisher information matrix for the logistic model.

_Logistic\_Jacobian.jl_

- Computes the Jacobian of the logistic model with respect to the parameters.

_Logistic\_partials.jl_

- Computes the partial derivatives of the logistic model with respect to the parameters, required to form Jacobian matrices.

_LogisticDataplot.jl_

- Produces plots of the logistic data and model

_LogisticSolAllParams.jl_

- Computes the analytical solution of the logistic model. 

_Logistic_data.csv_

- Contains coral area coverage data from the Australian Institute of Marine Science Long-term Monitoring Program, as presented in: MJ Simpson, AP Browning, DJ Warne, OJ Maclaren, RE Baker (2021). Parameter identifiability and model selection for sigmoid population growth models. _Journal of Theoretical Biology_.  


#### SIR model ####
Contains code for reproducing the inference and information geometry results for the susceptible, infected, recovered (SIR) model, presented in Figures 9 and 10 of the Sharp et al. paper. 

**The user callable scripts include:**

_SIR\_Infer\_beta\_gamma\_obsIonly.jl_

- Reproduces inference and information geometry results for the SIR model where only the infected population is observed, estimating unknown parameters β and γ.

_SIR\_Infer\_beta\_gamma.jl_

- Reproduces inference and information geometry results for the SIR model where all three populations are observed, estimating unknown parameters β and γ.

_SIR\_Infer\_beta\_StDev\_obsIonly.jl_

- Reproduces inference and information geometry results for the SIR model where only the infected population is observed, estimating unknown parameter β and the standard deviation of the data.

_SIR\_Infer\_beta\_StDev.jl_

- Reproduces inference and information geometry results for the SIR model where all three populations are observed, estimating unknown parameter β and the standard deviation of the data.

The runtime of each of these scripts is on the order of a few hours.

**The supporting functions and data are:**

_DataGenerateSIR_obsIonly.jl_

- Generates synthetic data for the SIR model where only the infected population is observed. 

_DataGenerateSIR.jl_

- Generates synthetic data for the SIR model where all three populations are observed.

_FIM\_Obs\_Process\_SIR\_obsIonly.jl_

- Computes the Fisher information matrix for the observation process of the SIR model where only the infected population is observed.

_FIM\_Obs\_Process\_SIR.jl_

- Computes the Fisher information matrix for the observation process of the SIR model where all three populations are observed.

_FisherSIR\_obsIonly.jl_

- Computes the Fisher information matrix for the observation process of the SIR model where only the infected population is observed.

_FisherSIR.jl_

- Computes the Fisher information matrix for the SIR model, based on the observation process and the model Jacobian.

_SIR\_Jacobian\_obsIonly.jl_

- Computes the Jacobian of the SIR model with respect to the parameters, where only the infected population is observed.

_SIR\_Jacobian.jl_

- Computes the Jacobian of the SIR model with respect to the parameters, where all three populations are observed.

_SIR\_partials.jl_

- Computes the partial derivatives of the SIR model with respect to the parameters, required to form Jacobian matrices.

_SIRdataplot.jl_

- Produces plots of the SIR data and model

_SIRODE.jl_

- Contains a function for the right hand side of the SIR model ODEs. 

_SIRSolAllParams.jl_

- Computes the approximate numerical solution of the SIR model. 

_MurrayData.csv_

- Contains influenza outbreak data from _Murray JD. 2002 Mathematical Biology I: an Introduction, 3rd ed. Heidelberg: Springer_.  

## Key packages ##

Key Julia packages used in this work include DifferentialEquations.jl (Rackauckas, 2017), Distributions.jl (Besancon, 2021), FiniteDifferences.jl, ForwardDiff.jl (Revels, 2016), LinearAlgebra.jl and NLopt.jl with the BOBYQA algorithm (Powell, 2009). 

Besançon M, Papamarkou T, Anthoff D, Arslan A, Byrne S, Lin D, Pearson J. 2021 Distributions.jl: Definition and Modeling of Probability Distributions in the JuliaStats Ecosystem. _Journal of Statistical Software_ **98**, 1--30. (doi.org/10.18637/jss.v098.i16).  
  
Powell MJD. 2009 The BOBYQA algorithm for bound constrained optimization without derivatives. Technical report. _Department of Applied Mathematics and Theoretical Physics. Cambridge, England_.
	  
Rackauckas C, Nie Q. 2017 DifferentialEquations.jl – A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia. _The Journal of Open Research Software_ **5**, 1--10. (doi.org/10.5334/jors.151). 
  
Revels J, Lubin M, Papamarkou T. 2016 Forward-mode automatic differentiation in Julia. _arXiv [Mathematical Software]_, 1--4. (https://arxiv.org/pdf/1607.07892.pdf). 

## Reproducibility ##

Results in the manuscript are produced using Julia v1.5.4. Data is generated using pseudo-random numbers; the stream of generated numbers may differ following a Julia version change (https://docs.julialang.org/en/v1.5/stdlib/Random/), regardless of the set random seed. For reproducibility we provide the origianally generated data from which the results are produced, however this is only necessary for identical reproduction of the results in the manuscript, and similar results can be generated without loading the original data. To run the scripts using the original data, replace the DataGenerate...() line with code that loads the appropriate data, for example in the script SIR_Infer_beta_gamma.jl:  


replace

_Ntime,Ndata = DataGenerateSIR([β,γ,S₀,I₀,R₀],TimePoints,NperT,σ) #generate synthetic data_

with

_using DelimitedFiles_

_Data = readdlm("SIRData.csv",',',Float64)_

_Ntime = vec(Data[:,1])_

_Ndata = Data[:,2:4]_


It is also necessary to ensure that the values for _TimePoints_, and _NperT_ are consistent with the loaded data. For example, when loading the data from LogisticData_earlymid.csv, ensure that _TimePoints = [1000/365.25,2500/365.25]_, and _NperT = [15,15]_, and when loading LogisticData_earlymidlate.csv, _TimePoints = [1000/365.25,2500/365.25,4000/365.25]_, and _NperT = [10,10,10]_. 
 
