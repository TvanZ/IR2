# Overview #

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

### Weekly Questions
For our upcoming advisor meeting, we would like to discuss: <br> 
<br> 
      (1.) When we discretize the generator’s probability as either a zero or one using the methodology disccused in 
      Paper 6, we sometimes can values that aren’t zero or one. (Assuming we correctly implemented the author's 
      discretization method.) Is this something we need to correct/take some measure against? Or, will the 
      discriminator just learn that these clicks are clearly fake and push the generator to produce strictly binary clicks?
<br> 
      (2.) Are our regression-based EM algorithm is underestimation position bias; however it looks as though the original
      paper's EM algorithm overestimated position bias. Could this be a result of us using different parameters, or 
      is there an error in our implementation?

### Paper

https://www.overleaf.com/1649798953dqxcrhhcqvgs
