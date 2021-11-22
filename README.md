Julia implementation of likelihood based inference and information geometry techniques, including geodesic curves and scalar curvature. This code is used to generate results by Sharp et al. in:

Sharp JA, Browning AP, Mapder T, Baker CM, Burrage K, Simpson MJ. 2021 Parameter estimation and uncertainty quantification using information geometry. _bioRxiv preprint_, 110277. (LINK).

**Linear and exponential growth models**
Contains code for reproducing the inference and information geometry results for the linear and exponential growth models. 

**Logistic growth model**
Contains code for reproducing the inference and information geometry results for the logistic growth model. 

##SIR model##
Contains code for reproducing the inference and information geometry results for the susceptible, infected, recovered (SIR) model. 
#### test ####
**ConfidenceRegions.jl**

**EquidistantCircumferencePoints.jl**

**InfoGeo2D.jl**

**MvNormalModel.jl**

**NormalModel.jl**

**optimise.jl**

##References

Key Julia packages used in this work include DifferentialEquations.jl (Rackauckas, 2017), Distributions.jl (Besancon, 2021), FiniteDifferences.jl, ForwardDiff.jl (Revels, 2016), LinearAlgebra.jl and NLopt.jl with the BOBYQA algorithm (Powell, 2009). 

Besan\c{c}on M, Papamarkou T, Anthoff D, Arslan A, Byrne S, Lin D, Pearson J. 2021 Distributions.jl: Definition and Modeling of Probability Distributions in the JuliaStats Ecosystem. \textit{Journal of Statistical Software} \textbf{98}, 1--30. (doi.org/10.18637/jss.v098.i16).  
  
Powell MJD. 2009 The BOBYQA algorithm for bound constrained optimization without derivatives. Technical report. \textit{Department of Applied Mathematics and Theoretical Physics. Cambridge, England}.
	  
Rackauckas C, Nie Q. 2017 DifferentialEquations.jl – A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia. \textit{The Journal of Open Research Software} \textbf{5}, 1--10. (doi.org/10.5334/jors.151). 
  
Revels J, Lubin M, Papamarkou T. 2016 Forward-mode automatic differentiation in Julia. \textit{arXiv [Mathematical Software]}, 1--4. (https://arxiv.org/pdf/1607.07892.pdf). 
