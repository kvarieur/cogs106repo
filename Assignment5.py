import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
import unittest
import numpy as np


class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def h_rate(self):
        return self.hits / (self.hits + self.misses)

    def f_alarm(self):
        return self.false_alarms / (self.false_alarms + self.correct_rejections)

    def d_prime(self):
        return norm.ppf(self.h_rate()) - norm.ppf(self.f_alarm())

    def criterion(self):
        return -0.5 * (norm.ppf(self.h_rate()) + norm.ppf(self.f_alarm()))

    def __add__(self, other):
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_rejections + other.correct_rejections
        )

    def __mul__(self, scalar):
        return SignalDetection(
            self.hits * scalar,
            self.misses * scalar,
            self.false_alarms * scalar,
            self.correct_rejections * scalar
        )

    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = []
        for i in range(len(criteriaList)):
            k = criteriaList[i] + (dprime / 2)
            hr = 1 - norm.cdf(k - dprime)
            fa = 1 - norm.cdf(k)
            hits = np.random.binomial(signalCount, hr)
            misses = signalCount - hits
            false_alarms = np.random.binomial(noiseCount, fa)
            criteria = noiseCount - false_alarms
            print(hits, misses, false_alarms, criteria)
            new_sig = SignalDetection(hits, misses, false_alarms, criteria)
            sdtList.append(new_sig)
        return sdtList

    def plot_sdt(self, d_prime):
        x = np.linspace(-4, 4, 1000)
        y_Noise = norm.pdf(x, loc=0, scale=1)
        y_Signal = norm.pdf(x, loc=d_prime, scale=1)
        c = d_prime / 2
        Noisetop_y = np.max(y_Noise)
        Noisestop_x = x[np.argmax(y_Noise)]
        Signaltop_y = np.max(y_Signal)
        Signaltop_x = x[np.argmax(y_Signal)]
        plt.plot(x, y_Noise, label="Noise")
        plt.plot(x, y_Signal, label="Signal")
        plt.axvline((d_prime / 2) + c, label='threshold', color='k', linestyle='--')
        plt.plot([Noisestop_x, Signaltop_x], [Noisetop_y, Signaltop_y], label="d'", linestyle='-')
        plt.ylim(ymin=0)
        plt.xlabel('Decision Variable')
        plt.ylabel('Probability')
        plt.title('Signal detection theory')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_roc(sdtList):
        plt.figure()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic Curve")
        if isinstance(sdtList, list):
            for i in range(len(sdtList)):
                sdt = sdtList[i]
                plt.plot(sdt.f_alarm(), sdt.h_rate(), 'o', color='black')
        x, y = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
        plt.plot(x, y, '--', color='black')
        plt.grid()

    @staticmethod
    def rocCurve(falseAlarmRate, a):
        return norm.cdf(a + norm.ppf(falseAlarmRate))

    def nLogLikelihood(self, hit_rate, false_alarm_rate):
        return -((self.hits * np.log(hit_rate)) +
        (self.misses * np.log(1-hit_rate)) +
        (self.false_alarms * np.log(false_alarm_rate)) +
        (self.correct_rejections * np.log(1-false_alarm_rate)))


    @staticmethod
    def fit_roc(sdtList):
        a = 0
        for sdt in sdtList:
            SignalDetection.plot_roc(sdtList)
            minimize = scipy.optimize.minimize(fun=sdt.rocLoss, x0=a, method='BFGS', args=sdtList)
            aHat = minimize.x
            x = np.linspace(0, 1, num=100)
            y = sdt.rocCurve(x, aHat)
            plt.plot(x, y, 'r-', limewidth=2, markersize=8)
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.legend()
        plt.title('Receive Operating Charateristic')
        return float(aHat)

    @staticmethod
    def rocLoss(a, sdtList):
        Loss = 0
        for sdt in sdtList:
            phr = sdt.rocCurve(sdt.f_alarm(), a)
            lsum = sdt.nLogLikelihood(phr, sdt.f_alarm())
            Loss = Loss + lsum
        return Loss


sdtList = SignalDetection.simulate(1, [-1, 0, 1], 1e7, 1e7)
SignalDetection.fit_roc(sdtList)
plt.show()

#unit test from Professor V.
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
