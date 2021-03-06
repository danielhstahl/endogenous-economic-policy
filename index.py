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
    return eta * l1_curr + eta * l2_curr - l2_drift + eta * l3_curr - l3_drift


def drift_l2(
    gamma0: float,
    gamma1: float,
    gamma2: float,
    deviation_from_natural_interest_rate: float,  # i*-i
    gdp_ratio: float,
    l2_curr: float,
) -> float:
    return (
        gamma0 + gamma1 * deviation_from_natural_interest_rate + gamma2 * gdp_ratio
    ) * l2_curr


# drift is constrained by fraction of output
def drift_l3(v0: float, v1: float, gdp_ratio: float, l3_curr: float, y: float) -> float:
    den = y * gdp_ratio
    return (v0 + v1 * gdp_ratio) * l3_curr * (1.0 - l3_curr / den) if den > 0.0 else 0.0


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


def drift_rate(a: float, natural_rate: float, rate_curr: float) -> float:
    return a * (natural_rate - rate_curr)


class Economy:
    def __init__(self, y, k, a, shock_rate, l1, l2, l3):
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
        self.rate = [shock_rate]


class EvolveEconomy:
    def __init__(
        self,
        eta,
        gamma0,
        gamma1,
        gamma2,
        v0,
        v1,
        a,
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
        self.a = a
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

    def execute_period(
        self,
        dt: float,
        natural_rate: float,
        # deviation_from_natural_interest_rate: float,
        gdp_ratio: float,
        econ: Economy,
    ):
        l2_drift = drift_l2(
            self.gamma0,
            self.gamma1,
            self.gamma2,
            natural_rate - econ.rate[-1],
            gdp_ratio,
            econ.l2[-1],
        )
        l3_drift = drift_l3(self.v0, self.v1, gdp_ratio, econ.l3[-1], econ.y[-1])
        l1_drift = drift_l1(
            self.eta, econ.l1[-1], econ.l2[-1], econ.l3[-1], l2_drift, l3_drift
        )
        rate_drift = drift_rate(self.a, natural_rate, econ.rate[-1])
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
        curr_l2 = max(l2_drift * dt + econ.l2[-1], 0)
        curr_l3 = max(l3_drift * dt + econ.l3[-1], 0)

        curr_k = max(econ.k[-1] * k_drift * dt + econ.k[-1], 0)
        curr_a = a_drift * dt + econ.a[-1]
        curr_y = y(curr_k, curr_a, curr_l1, self.alpha)
        curr_l = curr_l1 + curr_l2 + curr_l3
        curr_rate = econ.rate[-1] + rate_drift * dt
        econ.l1.append(curr_l1)
        econ.l2.append(curr_l2)
        econ.l3.append(curr_l3)
        econ.k.append(curr_k)
        econ.a.append(curr_a)
        econ.y.append(curr_y)
        econ.rate.append(curr_rate)

        econ.y_per_labor.append(curr_y / curr_l)
        econ.total_l.append(curr_l)
        econ.t.append(econ.t[-1] + dt)

    def simulate(
        self,
        # deviation_from_natural_interest_rate: float,
        natural_rate: float,
        shock_rate: float,
        gdp_ratio: float,
        t: float,
        n: int,
    ) -> Economy:
        econ = Economy(
            y=self.y_init,
            k=self.k_init,
            a=self.a_init,
            shock_rate=shock_rate,
            l1=self.l1_init,
            l2=self.l2_init,
            l3=self.l3_init,
        )
        dt = t / (n)
        for i in range(n):
            self.execute_period(dt, natural_rate, gdp_ratio, econ)
        return econ


if __name__ == "__main__":
    economy_parameters = [
        {
            "eta": 0.02,
            "gamma0": 0.004,
            "gamma1": 1.5,
            "gamma2": 0.005,  # small amount of spending on R&D
            "v0": 0.001,  # Low rate of bureacracy with no govt spending
            "v1": 0.05,
            "a": 0.1,
            "beta": 0.03,
            "alpha": 0.4,
            "delta": 0.1,
            "s": 0.3,
            "l1_init": 3.0,
            "l2_init": 1.0,
            "l3_init": 1.0,
            "a_init": 2.0,
            "k_init": 10.0,
        },
        {
            "eta": 0.02,
            "gamma0": 0.02,  # large enough to cause L2 to explode relative to l1 and l3
            "gamma1": 1.5,  # this should increase L2...right?
            "gamma2": 0.005,  # small amount of spending on R&D
            "v0": 0.001,  # Low rate of bureacracy with no govt spending
            "v1": 0.05,
            "a": 0.1,
            "beta": 0.03,
            "alpha": 0.4,
            "delta": 0.1,
            "s": 0.3,
            "l1_init": 3.0,
            "l2_init": 1.0,
            "l3_init": 1.0,
            "a_init": 2.0,
            "k_init": 10.0,
        },
    ]
    example_inputs = [
        (0.08, 0.05, 0.15),
        (0.08, 0.02, 0.15),
        (0.08, 0.05, 0.0),
        (0.08, 0.1, 0.0),
        (0.08, 0.08, 0.15),
        (0.08, 0.15, 0.15),
        (0.08, 0.05, 0.8),
    ]
    t = 100
    n = 100000
    for index, economy in enumerate(economy_parameters):
        econ = EvolveEconomy(**economy)
        for sub_index, (natural_rate, rate_shock, ratio) in enumerate(example_inputs):
            sim = econ.simulate(
                natural_rate=natural_rate,
                shock_rate=rate_shock,
                gdp_ratio=ratio,
                t=t,
                n=n,
            )
            plt.figure(
                f"econ_{index}_run_main"
            )  # this needs to aggregate plots across rate scenarios
            plt.plot(
                sim.t,
                sim.y_per_labor,
                label=f"rate shock={rate_shock}, gdp_ratio={ratio}",
            )
            plt.figure(f"econ_{index}_run_{sub_index}_labor")
            plt.plot(
                sim.t,
                sim.l1,
                label=f"L1",
            )
            plt.plot(
                sim.t,
                sim.l2,
                label=f"L2",
            )
            plt.plot(
                sim.t,
                sim.l3,
                label=f"L3",
            )
            plt.legend(loc="upper left")
            plt.title(f"Labor for rate shock={rate_shock}, gdp_ratio={ratio}")
            plt.savefig(f"images/econ_{index}_run_{sub_index}_labor.eps", format="eps")
        plt.figure(f"econ_{index}_run_main")
        plt.title(f"Output per capita")
        plt.legend(loc="upper left")
        plt.savefig(f"images/economy_{index}.eps", format="eps")
        # plt.savefig(f"output_per_labor_{t}.png")
        # plt.clf()
