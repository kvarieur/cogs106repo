import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class SignalDetection:

    def __init__(self, hits, misses, falseAlarms, correctRejections):
        "Initializes the class using the self object and the signal detection variables."
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def __str__(self):
        "Returns the class as a labeled list to enable printing and error detection."
        return f"hits: {self.hits}, misses: {self.misses}, false alarms: {self.falseAlarms}, correct rejections: {self.correctRejections}"

    def hit_rate(self):
        "Returns the hit rate based on the class object."
        return self.hits / (self.hits+self.misses)

    def false_alarm_rate(self):
        "Returns the false alarm rate based on the class object."
        return  self.falseAlarms / (self.falseAlarms + self.correctRejections)

    def d_prime(self):
        "Returns the d-prime value given the hit rate and false alarm rate."
	return stats.normppf(self.hit_rate()) - stats.norm.ppf(self.false_alarm_rate())

    def criterion(self):
        "Returns the criterion value given the hit rate and false alarm rate."
        return -0.5 * stats.norm.ppf(hr)-stats.norm.ppf(far)

    # overloading the + and * methods
    def __add__(self, other):
        "add up each type of trial from two objects."
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.falseAlarms + other.falseAlarms,
            self.correctRejections + other.correctRejections)

    def __mul__(self, scalar):
        "multiply all types of trials with a scalar."
        return SignalDetection(
            self.hits * scalar,
            self.misses * scalar,
            self.falseAlarms * scalar,
            self.correctRejections * scalar)

    # adding ROC curve method
    def plot_roc(self):
        #variables to be used in plot
        far = self.false_alarm_rate()
        hr = self.hit_rate()
        #creating coordinates for false alarm rate and hit rate lines
        far_coords = [0.0, far, 1.0]
        hr_coords = [0.0, hr, 1.0]
        #make plots
        plt.plot(far_coords, hr_coords, 'bo-', label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k-', label='reference line')
        plt.xlim = ([0.0, 1.0])
        plt.ylim = ([0.0, 1.0])
        plt.ylabel('Hit Rate')
        plt.xlabel('False Alarm Rate')
        plt.legend()
        plt.title('Receive Operating Characteristic')
        plt.show()

    def plot_sdt(self, d_prime):
        c = d_prime / 2  # threshold value
        x = np.linspace(-4, 4, 1000)  # axes
        signal = stats.norm.pdf(x, loc=d_prime, scale=1) #signalcurve
        noise = stats.norm.pdf(x, loc=0, scale=1) #noisecurve

        # calculate max of signal and noise curves for d' line
        Nmax_y = np.max(noise)
        Nmax_x = x[np.argmax(noise)]
        Smax_y = np.max(signal)
        Smax_x = x[np.argmax(signal)]

        #plot curves
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

#call the plot functions with h,m,fa,cr
sd = SignalDetection(30, 60, 80, 70)
sd.plot_roc()
sd.plot_sdt(sd.d_prime())

#run the unit test suite
import unittest
from signaldetection import SignalDetection

class TestSignalDetection(unittest.TestCase):

    def test_d_prime_zero(self):
        sd = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_d_prime_nonzero(self):
        sd = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_zero(self):
        sd = SignalDetection(5, 5, 5, 5)
        # Calculate expected criterion
        expected = 0
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_nonzero(self):
        sd = SignalDetection(15, 10, 15, 5)
        # Calculate expected criterion
        expected = -0.463918426665941
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)

#test for corruption in code by altering hits
    def test_for_corruption(self):
        sd = SignalDetection(15, 5, 15, 5)
        obtained1 = sd.criterion()
        sd.hits = 9
        obtained2 = sd.criterion()
        self.assertNotEqual(obtained1, obtained2)

    def test_addition(self):
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)

    def test_multiplication(self):
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
