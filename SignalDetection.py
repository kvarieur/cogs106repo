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
        self.hr = (self.hits / (self.hits + self.misses))
        return self.hr

    def false_alarm_rate(self):
        "Returns the false alarm rate based on the class object."
        self.far = (self.falseAlarms / (self.falseAlarms + self.correctRejections))
        return self.far

    def d_prime(self):
        "Returns the d-prime value given the hit rate and false alarm rate."
        hr = self.hit_rate()
        far = self.false_alarm_rate()
        z_h = stats.norm.ppf(hr)
        z_fa = stats.norm.ppf(far)
        self.dp = z_h - z_fa
        return self.dp

    def criterion(self):
        "Returns the criterion value given the hit rate and false alarm rate."
        hr = self.hit_rate()
        far = self.false_alarm_rate()
        z_h = stats.norm.ppf(hr)
        z_fa = stats.norm.ppf(far)
        self.c = -0.5 * (z_h + z_fa)
        return self.c

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
        plt.plot([Nmax_x, Smax_x], [Nmax_y, Smax_y], linestyle='--', lw=2, color='b')
        plt.legend()
        plt.xlabel('Stimulus intensity')
        plt.ylabel('Probability density')
        plt.title('Signal Detection Theory')
        plt.show()

#call the plot functions with h,m,fa,cr
sd = SignalDetection(30, 60, 80, 70)
sd.plot_roc()
sd.plot_sdt(sd.d_prime())
