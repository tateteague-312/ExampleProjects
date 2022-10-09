
import numpy as np


def bootstrap(x, confidence=.68, nSamples=100):
    '''
    Make "nSamples" new datasets by re-sampling x with replacement
    the size of the samples should be the same as x itself
    '''
    means = []
    for k in range(nSamples):
        sample = np.random.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))
    means.sort()
    leftTail = int(((1.0 - confidence)/2) * nSamples)
    rightTail = (nSamples - 1) - leftTail
    return means[leftTail], np.mean(x), means[rightTail]


class MonteCarlo:
    """
    Chnaged results handling as well as convergence limit to end simulation once we had convergence for all of our measured metrics 
    """

    def SimulateOnce(self):
        raise NotImplementedError

    def RunSimulation(self, threshold=.001, simCount=100):
        self.con = [] 
        self.bet = [] 
        self.deg = []       # Array to hold the results
        sum1_con = 0.0              
        sum2_con = 0.0              
        sum1_bet = 0.0              
        sum2_bet = 0.0
        sum1_deg = 0.0              
        sum2_deg = 0.0


        # Now, we set up the simulation loop
        self.convergence = False
        # while self.convergence == False:
        for k in range(simCount):   
            x = self.SimulateOnce()     # Run the simulation
            self.con.append(x[0])
            self.bet.append(x[1])
            self.deg.append(x[2])                        # Add the result to the array
            sum1_con += x[0]                   # Add it to the sum
            sum2_con += x[0]**2                 # Add the square to the sum of squares
            sum1_bet += x[1]                   
            sum2_bet += x[1]**2                 
            sum1_deg += x[2]                   
            sum2_deg += x[2]**2                 

        # Go to at least a 100 cycles before testing for convergence.             
            if k > 100:
                mu_con = float(sum1_con)/k                  # Compute the mean
                var_con = (float(sum2_con)/(k-1)) - mu_con**2   # An alternate calculation of the variance
                dmu_con = np.sqrt(var_con / k)              # Standard error of the mean

                mu_bet = float(sum1_bet)/k                  
                var_bet = (float(sum2_bet)/(k-1)) - mu_bet**2   
                dmu_bet = np.sqrt(var_bet / k)      

                mu_deg = float(sum1_deg)/k                  
                var_deg = (float(sum2_deg)/(k-1)) - mu_deg**2   
                dmu_deg = np.sqrt(var_deg / k)
        # If the estimate of the error in mu is within "threshold" percent
        # then set convergence to true.  We could also break out early at this 
        # point if we wanted to 
                if dmu_con < abs(mu_con) * threshold:
                    self.convergence = True
                elif dmu_bet < abs(mu_bet) * threshold:
                    self.convergence = True
                elif dmu_deg < abs(mu_deg) * threshold:
                    self.convergence = True

    # Bootstrap the results and return not only the mean, but the confidence interval
    # as well.  [mean - se, mean, mean + se]
        return bootstrap(self.con), bootstrap(self.deg), bootstrap(self.bet)

            
