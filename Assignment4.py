import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt


class SignalDetection:

    def __init__(self, hits, misses, falseAlarms, correctRejections):
        """
        Initializes the class using the self object and the signal detection variables.
        """
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def __str__(self):
        """
        Returns the class as a labeled list to enable printing and error detection.
        """
        return f"hits: {self.hits}, misses: {self.misses}, false alarms: {self.falseAlarms}, correct rejections: {self.correctRejections}"

    def hit_rate(self):
        """
        Returns the hit rate based on the class object.
        """
        return self.hits / (self.hits + self.misses)

    def false_alarm_rate(self):
        """
        Returns the false alarm rate based on the class object.
        """
        return self.falseAlarms / (self.falseAlarms + self.correctRejections)

    def d_prime(self):
        """
        Returns the d-prime value given the hit rate and false alarm rate.
        """
        return stats.norm.ppf(self.hit_rate()) - stats.norm.ppf(self.false_alarm_rate())

    def criterion(self):
        """
        Returns the criterion value given the hit rate and false alarm rate.
        """
        return -0.5 * stats.norm.ppf(self.hit_rate())-stats.norm.ppf(self.false_alarm_rate())

    # overloading the + and * methods
    def __add__(self, other):
        """
        add up each type of trial from two objects.
        """
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.falseAlarms + other.falseAlarms,
            self.correctRejections + other.correctRejections)

    def __mul__(self, scalar):
        """
        multiply all types of trials with a scalar.
        """
        return SignalDetection(
            self.hits * scalar,
            self.misses * scalar,
            self.falseAlarms * scalar,
            self.correctRejections * scalar)

    @staticmethod
    def simulate(d_prime, criteriaList, signalCount, noiseCount):
        """
        simulate signal detection object 
        """
        sdtList = []
        for i in range(len(criteriaList)):
            criteria = criteriaList[i]
            k = criteria + d_prime / 2
            hr = 1 - stats.norm.cdf(k - d_prime)
            far = 1 - stats.norm.cdf(k)
            hits = np.random.binomial(n=signalCount, p=hr)
            misses = signalCount - hits
            falseAlarms = np.random.binomial(n=noiseCount, p=far) #get the hr and far from the dprime and criteria list and then use those to do random number generation for hits, misses, false alarms and correct rejections
            correctRejections = noiseCount - falseAlarms
            sdtList.append(SignalDetection(hits, misses, falseAlarms, correctRejections))
        return sdtList

    # adding ROC plot method
    @staticmethod
    def plot_roc(sdtList):
        plt.plot([0,1], [0,1], 'k--', label= 'Chance')
        #variables to be used in plot
        for sdt in sdtList:
            hr = sdt.hit_rate()
            far = sdt.false_alarm_rate()
            plt.plot(far, hr, 'o', label=f'd prime={sdt.d_prime():.2f}', linewidth=2, markersize=8)
	    plt.grid()
        plt.ylabel('Hit Rate')
        plt.xlabel('False Alarm Rate')
        plt.legend()
        plt.title('ROC Curve')
        plt.show()

    def nLogLikelihood(self, hit_rate, false_alarm_rate):
        return - ((self.hits * np.log(hit_rate)) +
                  (self.misses * np.log(1 - hit_rate)) +
                  (self.falseAlarms * np.log(false_alarm_rate)) +
                  (self.correctRejections * np.log(1 - false_alarm_rate)))

    @staticmethod
    def rocCurve(false_alarm_rate, a):
        return stats.norm.cdf(a + stats.norm.ppf(false_alarm_rate))

    @staticmethod
    def rocLoss(a, sdtList):
        Loss = 0
        for sdt in sdtList:
            far = sdt.false_alarm_rate()
            phr = sdt.rocCurve(far, a) #calculate predicted hit rate with above function
            lsum = sdt.nLogLikelihood(phr, far) #adding up losses of nll of predicted hr and sdt objects
            Loss += lsum
            return Loss

    @staticmethod
    def fit_roc(sdtList):
        a = 0
        for sdt in sdtList:
            SignalDetection.plot_roc(sdtList)
            minimize = optimize.minimize(fun=sdt.rocLoss, args=sdtList, x0=a, method='BFGS') #fitting the function: minimizing a
            aHat = minimize.x
            x = np.linspace(0, 1, num=100)
            y = sdt.rocCurve(x, aHat)
            plt.plot(x, y, 'r-', linewidth=2, markersize=8)
        plt.ylabel('Hit Rate')
        plt.xlabel('False Alarm Rate')
        plt.title('Receive Operating Characteristic')
        plt.legend()
        return float(aHat)
        #plt.show()

    def plot_sdt(self, d_prime):
        c = d_prime / 2  # threshold value
        x = np.linspace(-4, 4, 1000)  # axes
        signal = stats.norm.pdf(x, loc=d_prime, scale=1) #signalcurve
        noise = stats.norm.pdf(x, loc=0, scale=1) #noisecurve
        """
        calculate max of signal and noise curves for d' line
        """
        Nmax_y = np.max(noise)
        Nmax_x = x[np.argmax(noise)]
        Smax_y = np.max(signal)
        Smax_x = x[np.argmax(signal)]

        """
        plot curves
        """
        plt.plot(x, signal, label='Noise')
        plt.plot(x, noise, label='Signal')
        plt.axvline(x=(d_prime / 2) + c, color='g', linestyle='--',
                    label='Threshold')  # vertical line over plot for d'/2+c
        plt.plot([Nmax_x, Smax_x], [Nmax_y, Smax_y], linestyle='--', lw=2, color='r', label = 'd prime')
        plt.legend()
        plt.xlabel('Stimulus intensity')
        plt.ylabel('Probability density')
        plt.title('Signal Detection Theory')
        plt.show()


sdtList = SignalDetection.simulate(1.5, [-0.5, 0.0, 0.5, 0.7, 0.9], 100, 100) #hard coded test to check if plots are correct according to sd object
for sdt in sdtList:
    print(sdt)
#SignalDetection.plot_roc(sdtList)
SignalDetection.fit_roc(sdtList)

#unit test from Professor V
import numpy as np
import matplotlib.pyplot as plt
import unittest
from signaldetection import SignalDetection

class TestSignalDetection(unittest.TestCase):
        """
        Test suite for SignalDetection class.
        """
        def test_d_prime_zero(self):
            """
            Test d-prime calculation when hits and false alarms are 0.
            """
            sd = SignalDetection(15, 5, 15, 5)
            expected = 0
            obtained = sd.d_prime()
            self.assertAlmostEqual(obtained, expected, places=10)

        def test_d_prime_nonzero(self):
            """
            Test d-prime calculation when hits and false alarms are nonzero.
            """
            sd = SignalDetection(15, 10, 15, 5)
            expected = -0.421142647060282
            obtained = sd.d_prime()
            self.assertAlmostEqual(obtained, expected, places=10)

        def test_criterion_zero(self):
            """
            Test criterion calculation when hits and false alarms are both 0.
            """
            sd = SignalDetection(5, 5, 5, 5)
            expected = 0
            obtained = sd.criterion()
            self.assertAlmostEqual(obtained, expected, places=10)

        def test_criterion_nonzero(self):
            """
            Test criterion calculation when hits and false alarms are nonzero.
            """
            sd = SignalDetection(15, 10, 15, 5)
            expected = -0.463918426665941
            obtained = sd.criterion()
            self.assertAlmostEqual(obtained, expected, places=10)

        def test_addition(self):
            """
            Test addition of two SignalDetection objects.
            """
            sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
            expected = SignalDetection(3, 2, 3, 4).criterion()
            obtained = sd.criterion()
            self.assertEqual(obtained, expected)

        def test_multiplication(self):
            """
            Test multiplication of a SignalDetection object with a scalar.
            """
            sd = SignalDetection(1, 2, 3, 1) * 4
            expected = SignalDetection(4, 8, 12, 4).criterion()
            obtained = sd.criterion()
            self.assertEqual(obtained, expected)

        def test_simulate_single_criterion(self):
            """
            Test SignalDetection.simulate method with a single criterion value.
            """
            dPrime = 1.5
            criteriaList = [0]
            signalCount = 1000
            noiseCount = 1000

            sdtList = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
            self.assertEqual(len(sdtList), 1)
            sdt = sdtList[0]

            self.assertEqual(sdt.hits, sdtList[0].hits)
            self.assertEqual(sdt.misses, sdtList[0].misses)
            self.assertEqual(sdt.falseAlarms, sdtList[0].falseAlarms)
            self.assertEqual(sdt.correctRejections, sdtList[0].correctRejections)

        def test_simulate_multiple_criteria(self):
            """
            Test SignalDetection.simulate method with multiple criterion values.
            """
            dPrime = 1.5
            criteriaList = [-0.5, 0, 0.5]
            signalCount = 1000
            noiseCount = 1000
            sdtList = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
            self.assertEqual(len(sdtList), 3)
            for sdt in sdtList:
                self.assertLessEqual(sdt.hits, signalCount)
                self.assertLessEqual(sdt.misses, signalCount)
                self.assertLessEqual(sdt.falseAlarms, noiseCount)
                self.assertLessEqual(sdt.correctRejections, noiseCount)

        def test_nLogLikelihood(self):
            """
            Test case to verify nLogLikelihood calculation for a SignalDetection object.
            """
            sdt = SignalDetection(10, 5, 3, 12)
            hit_rate = 0.5
            false_alarm_rate = 0.2
            expected_nll = - (10 * np.log(hit_rate) +
                              5 * np.log(1 - hit_rate) +
                              3 * np.log(false_alarm_rate) +
                              12 * np.log(1 - false_alarm_rate))
            self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                                   expected_nll, places=6)

        def test_rocLoss(self):
            """
            Test case to verify rocLoss calculation for a list of SignalDetection objects.
            """
            sdtList = [
                SignalDetection(8, 2, 1, 9),
                SignalDetection(14, 1, 2, 8),
                SignalDetection(10, 3, 1, 9),
                SignalDetection(11, 2, 2, 8),
            ]
            a = 0
            expected = 99.3884
            self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)

        def test_integration(self):
            """
            Test case to verify integration of SignalDetection simulation and ROC fitting.
            """
            dPrime = 1
            sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
            aHat = SignalDetection.fit_roc(sdtList)
            self.assertAlmostEqual(aHat, dPrime, places=2)
            plt.close()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
