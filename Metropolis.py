import numpy as np
from signaldetection import SignalDetection

class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.current = initialState
        self.samples = []
        self.sd = 1

    def __accept(self, proposal):
        """A private method that checks whether to accept or reject the proposed value proposal based on the acceptance probability calculated from the current state and the proposed state. It returns True if the proposal is accepted and False otherwise."""
        logAcceptanceProb = min(0, (self.logTarget(proposal) - self.logTarget(self.current))) #convert to log
        if np.log(np.random.uniform()) < logAcceptanceProb:
            self.current = proposal
            return True
        else:
            return False

    def adapt(self, blockLengths):
        for k in blockLengths:
            acceptances = 0 
            for n in range(k):
                proposal = np.random.normal(self.current, self.sd)
                if self.__accept(proposal):
                    acceptances += 1
            acceptanceRate = acceptances / n #computes acceptance rate rk
            self.sd *= (acceptanceRate/0.4)**1.1
        return self

    def sample(self, n):
        for i in range(n):
            proposal = np.random.normal(self.current, self.sd)
            accept = self.__accept(proposal)
            if accept:
                self.current = proposal
            self.samples.append(self.current) #appending the samples into the sample array to create a chain of states convering to the target distribution
        return self

    def summary(self):
        samples = np.array(self.samples)
        mean    = np.mean(self.samples)
        c025    = np.percentile(self.samples, 2.5)
        c975    = np.percentile(self.samples, 97.)
        return {'mean': mean, 'c025': c025, 'c975': c975}

#Integration test
import scipy.stats
import scipy
import matplotlib.pyplot as plt
import numpy as np

from signaldetection import SignalDetection
from Metropolis import Metropolis

def fit_roc_bayesian(sdtList):

    # Define the log-likelihood function to optimize
    def loglik(a):
        return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc = 0, scale = 10)

    # Create a Metropolis sampler object and adapt it to the target distribution
    sampler = Metropolis(logTarget=loglik, initialState=0)
    sampler = sampler.adapt(blockLengths=[2000]*3) #2000 is n and 3 is k (3 blocks of 2000 iterations

    # Sample from the target distribution
    sampler = sampler.sample(n=4000)

    # Compute the summary statistics of the samples
    result = sampler.summary()

    # Print the estimated value of the parameter a and its credible interval
    print(f"Estimated a: {result['mean']} ({result['c025']}, {result['c975']})")

    # Create a mosaic plot with four subplots
    fig, axes = plt.subplot_mosaic(
        [["ROC curve", "ROC curve", "traceplot"],
         ["ROC curve", "ROC curve", "histogram"]],
        constrained_layout=True
    )

    # Plot the ROC curve of the SDT data
    plt.sca(axes["ROC curve"])
    SignalDetection.plot_roc(sdtList=sdtList)

    # Compute the ROC curve for the estimated value of a and plot it
    xaxis = np.arange(start=0.00,
                      stop=1.00,
                      step=0.01)

    plt.plot(xaxis, SignalDetection.rocCurve(xaxis, result['mean']), 'r-')

    # Shade the area between the lower and upper bounds of the credible interval
    plt.fill_between(x=xaxis,
                     y1=SignalDetection.rocCurve(xaxis, result['c025']),
                     y2=SignalDetection.rocCurve(xaxis, result['c975']),
                     facecolor='r',
                     alpha=0.1)

    # Plot the trace of the sampler
    plt.sca(axes["traceplot"])
    plt.plot(sampler.samples)
    plt.xlabel('iteration')
    plt.ylabel('a')
    plt.title('Trace plot')

    # Plot the histogram of the samples
    plt.sca(axes["histogram"])
    plt.hist(sampler.samples,
             bins    = 51,
             density = True)
    plt.xlabel('a')
    plt.ylabel('density')
    plt.title('Histogram')

    # Show the plot
    plt.show()

# Define the number of SDT trials and generate a simulated dataset
sdtList = SignalDetection.simulate(d_prime      = 1,
                                   criteriaList = [-1, 0, 1],
                                   signalCount  = 40,
                                   noiseCount   = 40)
# Fit the ROC curve to the simulated dataset
fit_roc_bayesian(sdtList)
