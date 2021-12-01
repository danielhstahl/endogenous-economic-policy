from math import pow
import matplotlib.pyplot as plt


def drift_l1(
    eta: float,
    l1_curr: float,
    l2_curr: float,
    l3_curr: float,
    l2_drift: float,
    l3_drift: float,
) -> float:
    return eta * l1_curr + (eta - l2_drift) * l2_curr + (eta - l3_drift) * l3_curr


def drift_l2(
    gamma0: float, gamma1: float, gamma2: float, interest_rate: float, gdp_ratio: float
) -> float:
    return gamma0 - gamma1 * interest_rate + gamma2 * gdp_ratio


def drift_l3(v0: float, v1: float, gdp_ratio: float) -> float:
    return v0 + v1 * gdp_ratio


def drift_a(l2_curr: float, beta: float) -> float:
    return beta * l2_curr


def y(k_curr: float, a_curr: float, l1_curr: float, alpha: float) -> float:
    return pow(k_curr, alpha) * pow(a_curr * l1_curr, 1.0 - alpha)


def drift_k(
    k_curr: float,
    a_curr: float,
    l1_curr: float,
    l2_curr: float,
    s: float,
    delta: float,
    alpha: float,
) -> float:
    return s * y(k_curr, a_curr, l1_curr, alpha) - l2_curr - delta * k_curr


class Economy:
    def __init__(self, y, k, a, l1, l2, l3):
        total_l = l1 + l2 + l3
        self.y = [y]
        self.k = [k]
        self.a = [a]
        self.l1 = [l1]
        self.l2 = [l2]
        self.l3 = [l3]
        self.y_per_labor = [y / total_l]
        self.t = [0]
        self.total_l = [total_l]


class EvolveEconomy:
    def __init__(
        self,
        eta,
        gamma0,
        gamma1,
        gamma2,
        v0,
        v1,
        beta,
        alpha,
        delta,
        s,
        l1_init,
        l2_init,
        l3_init,
        a_init,
        k_init,
    ):
        self.eta = eta
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.v0 = v0
        self.v1 = v1
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.s = s
        self.l1_init = l1_init
        self.l2_init = l2_init
        self.l3_init = l3_init
        self.a_init = a_init
        self.k_init = k_init
        self.y_init = y(k_init, a_init, l1_init, alpha)
        # self.l1 = [l1_init]
        # self.l2 = [l2_init]
        # self.l3 = [l3_init]
        # self.a = [a_init]
        # self.k = [k_init]
        # self.y = [y(k_init, a_init, l1_init, alpha)]
        # TODO do I need to assert bounds on l2 and l3 drift?

    def execute_period(
        self, dt: float, interest_rate: float, gdp_ratio: float, econ: Economy
    ):
        l2_drift = drift_l2(
            self.gamma0, self.gamma1, self.gamma2, interest_rate, gdp_ratio
        )
        l3_drift = drift_l3(self.v0, self.v1, gdp_ratio)
        l1_drift = drift_l1(
            self.eta, econ.l1[-1], econ.l2[-1], econ.l3[-1], l2_drift, l3_drift
        )
        k_drift = drift_k(
            econ.k[-1],
            econ.a[-1],
            econ.l1[-1],
            econ.l2[-1],
            self.s,
            self.delta,
            self.alpha,
        )
        a_drift = drift_a(econ.l2[-1], self.beta)
        curr_l1 = max(l1_drift * dt + econ.l1[-1], 0)
        curr_l2 = max(econ.l2[-1] * l2_drift * dt + econ.l2[-1], 0)
        curr_l3 = max(econ.l3[-1] * l3_drift * dt + econ.l3[-1], 0)

        curr_k = max(econ.k[-1] * k_drift * dt + econ.k[-1], 0)
        curr_a = a_drift * dt + econ.a[-1]
        curr_y = y(curr_k, curr_a, curr_l1, self.alpha)
        curr_l = curr_l1 + curr_l2 + curr_l3
        print("kdrift", k_drift)
        # if curr_k > 0:
        # print(
        #     "l2",
        #    curr_l2,
        #    "l1",
        #    curr_l1,
        #    "k",
        #    curr_k,
        #    "kdrift",
        #    k_drift,
        #   "y",
        #   curr_y,
        # )

        econ.l1.append(curr_l1)
        econ.l2.append(curr_l2)
        econ.l3.append(curr_l3)
        econ.k.append(curr_k)
        econ.a.append(curr_a)
        econ.y.append(curr_y)

        econ.y_per_labor.append(curr_y / curr_l)
        econ.total_l.append(curr_l)
        econ.t.append(econ.t[-1] + dt)

    def simulate(
        self, interest_rate: float, gdp_ratio: float, t: float, n: int
    ) -> Economy:
        econ = Economy(
            y=self.y_init,
            k=self.k_init,
            a=self.a_init,
            l1=self.l1_init,
            l2=self.l2_init,
            l3=self.l3_init,
        )
        dt = t / (n)
        for i in range(n):
            self.execute_period(dt, interest_rate, gdp_ratio, econ)
        return econ


if __name__ == "__main__":
    economy = EvolveEconomy(
        eta=0.02,
        gamma0=0.02,
        gamma1=0.2,
        gamma2=0.005,  # small amount of spending on R&D
        v0=0.001,  # Low rate of bureacracy with no govt spending
        v1=0.05,
        beta=0.03,
        alpha=0.4,
        delta=0.1,
        s=0.3,
        l1_init=3.0,
        l2_init=1.0,
        l3_init=1.0,
        a_init=2.0,
        k_init=10.0,
    )
    t = 100
    n = 100000
    example_inputs = [
        (0.05, 0.15),
        (0.02, 0.15),
        (0.05, 0.0),
        (0.15, 0.15),
        (0.05, 0.8),
    ]
    for rate, ratio in example_inputs:
        sim = economy.simulate(interest_rate=rate, gdp_ratio=ratio, t=t, n=n)
        plt.plot(sim.t, sim.y_per_labor, label=f"rate={rate}, gdp_ratio={ratio}")
        # plt.plot(sim.t, sim.l1, label="l1")
        # plt.plot(sim.t, sim.l2, label="l2")
        # plt.plot(sim.t, sim.l3, label="l3")
        # plt.plot(sim.t, sim.a, label="a")
    plt.legend(loc="upper left")
    plt.savefig(f"output_per_labor_{t}.png")
