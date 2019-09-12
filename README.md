#Overview #

Position bias estimation is vital for counterfactual unbiased 
learning to rank. It estimates the probability that an item was observed in 
a ranking, however, only what is clicked is known. Under the assumption 
that a clicked item is observed, these models infer position bias models that 
try to predict observance.

Existing approaches use randomization experiments, EM-optimization or dual objective learning tasks. 
A recent Msc. A.I. thesis showed you can optimize click models using a GAN based objective, 
this strongly suggests that such an approach could also work for position bias estimation. 
Key here is that differentiable approximations of binary variables are used, 
recently a new method for such variables had also been introduced.


### Project



### Paper

https://www.overleaf.com/8179157659xrpbmnkxfzzk
