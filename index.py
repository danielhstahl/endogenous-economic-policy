from math import pow, exp
import matplotlib.pyplot as plt


##### new functions
def Y(K: float, A: float, L: float, V3: float, alpha: float) -> float:
    return pow(K, alpha) * pow(A * L * V3, 1.0 - alpha)


def drift_K(
    K: float,
    A: float,
    L: float,
    V1: float,
    V3: float,
    s: float,
    delta: float,
    alpha: float,
) -> float:
    return s * Y(K, A, L, V3, alpha) - L * V1 - delta * K


def drift_A(beta: float, L: float, V1: float) -> float:
    return beta * L * V1


def l1(
    gamma0: float,
    gamma1: float,
    gamma2: float,
    current_rate_deviation: float,
    current_govt: float,
) -> float:
    return exp(gamma0 + gamma1 * current_rate_deviation + gamma2 * current_govt)


def l2(psi0: float, psi1: float, current_govt: float) -> float:
    return exp(psi0 + psi1 * current_govt)


def get_current_rate_deviation(
    long_run_rate: float, shock_rate: float, a: float, t: float
) -> float:
    return (long_run_rate - shock_rate) * exp(-a * t)


def get_current_govt(
    long_run_govt: float, original_govt: float, b: float, t: float
) -> float:
    exp_b = exp(-b * t)
    return original_govt * exp_b + long_run_govt * (1 - exp_b)


# proportion of labor for technology
def V1(l1: float, l2: float) -> float:
    return l1 / (1.0 + l1 + l2)


# proportion of labor for administration
def V2(l1: float, l2: float) -> float:
    return l2 / (1.0 + l1 + l2)


# proportion of "traditional" labor
def V3(l1: float, l2: float) -> float:
    return 1.0 / (1.0 + l1 + l2)


def L(eta: float, init_L: float, t: float) -> float:
    return init_L * exp(eta * t)


####


class Economy:
    def __init__(self, y, k, a, l, v1, v2, v3):

        self.y = [y]
        self.k = [k]
        self.a = [a]
        self.l = [l]
        self.v1 = [v1]
        self.v2 = [v2]
        self.v3 = [v3]
        self.y_per_labor = [y / l]
        self.t = [0]


class EvolveEconomy:
    def __init__(
        self,
        eta,
        gamma0,
        gamma1,
        gamma2,
        psi0,
        psi1,
        a,
        b,
        beta,
        alpha,
        delta,
        s,
        l_init,
        a_init,
        k_init,
        natural_rate,
        shock_rate,
        original_govt,
        long_run_govt,
    ):
        self.natural_rate = natural_rate
        self.shock_rate = shock_rate
        self.original_govt = original_govt
        self.long_run_govt = long_run_govt
        curr_gov = get_current_govt(long_run_govt, original_govt, b, 0)
        init_l1 = l1(
            gamma0,
            gamma1,
            gamma2,
            get_current_rate_deviation(natural_rate, shock_rate, a, 0),
            curr_gov,
        )
        init_l2 = l2(psi0, psi1, curr_gov)

        init_V1 = V1(init_l1, init_l2)
        init_V2 = V2(init_l1, init_l2)
        init_V3 = V3(init_l1, init_l2)
        # V3(init_l1*

        self.eta = eta
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.psi0 = psi0
        self.psi1 = psi1
        self.a = a
        self.b = b
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.s = s
        self.l_init = l_init
        self.a_init = a_init
        self.k_init = k_init
        self.v1_init = init_V1
        self.v2_init = init_V2
        self.v3_init = init_V3
        self.y_init = Y(k_init, a_init, l_init, init_V3, alpha)

    def execute_period(
        self,
        dt: float,
        # natural_rate: float,
        # shock_rate: float,
        # gdp_ratio: float,
        econ: Economy,
    ):
        t = econ.t[-1] + dt
        interest_rate_deviation = get_current_rate_deviation(
            self.natural_rate, self.shock_rate, self.a, t
        )
        government_rate = get_current_govt(
            self.long_run_govt, self.original_govt, self.b, t
        )

        curr_l1 = l1(
            self.gamma0,
            self.gamma1,
            self.gamma2,
            interest_rate_deviation,
            government_rate,
        )
        curr_l2 = l2(self.psi0, self.psi1, government_rate)

        curr_V1 = V1(curr_l1, curr_l2)

        curr_V2 = V2(curr_l1, curr_l2)

        curr_V3 = V3(curr_l1, curr_l2)

        curr_L = L(self.eta, self.l_init, t)

        curr_a = econ.a[-1] + drift_A(self.beta, econ.l[-1], econ.v1[-1]) * dt

        curr_k = (
            econ.k[-1]
            + drift_K(
                econ.k[-1],
                econ.a[-1],
                econ.l[-1],
                econ.v1[-1],
                econ.v3[-1],
                self.s,
                self.delta,
                self.alpha,
            )
            * dt
        )

        econ.l.append(curr_L)
        econ.v1.append(curr_V1)
        econ.v2.append(curr_V2)
        econ.v3.append(curr_V3)
        econ.a.append(curr_a)
        curr_y = Y(curr_k, curr_a, curr_L, curr_V3, self.alpha)
        econ.y.append(curr_y)
        # econ.rate.append(interest_rate_deviation+)
        econ.y_per_labor.append(curr_y / curr_L)
        econ.t.append(econ.t[-1] + dt)

    def simulate(
        self,
        # deviation_from_natural_interest_rate: float,
        # natural_rate: float,
        # shock_rate: float,
        # gdp_ratio: float,
        t: float,
        n: int,
    ) -> Economy:  # y, k, a, l, v1, v2, v3, shock_rate, govt
        econ = Economy(
            y=self.y_init,
            k=self.k_init,
            a=self.a_init,
            l=self.l_init,
            v1=self.v1_init,
            v2=self.v2_init,
            v3=self.v3_init,
        )
        dt = t / (n)
        for i in range(n):
            self.execute_period(dt, econ)
        return econ


if __name__ == "__main__":
    economy_parameters = [
        {
            "eta": 0.02,
            "gamma0": 0.004,
            "gamma1": 3.0,
            "gamma2": 0.05,  # small amount of spending on R&D
            "psi0": 0.001,  # Low rate of bureacracy with no govt spending
            "psi1": 2.5,
            "a": 0.1,
            "b": 0.1,
            "beta": 0.03,
            "alpha": 0.4,
            "delta": 0.1,
            "s": 0.3,
            "l_init": 5.0,
            "a_init": 2.0,
            "k_init": 10.0,
            # "natural_rate": 0.08,
            # "shock_rate": 0.02,
            # "original_govt": 0.15,
            # "long_run_govt": 0.25
        }
    ]
    example_inputs = [
        (0.08, 0.05, 0.15, 0.15),
        (0.08, 0.02, 0.15, 0.15),
        (0.08, 0.05, 0.0, 0.15),
        (0.08, 0.1, 0.0, 0.15),
        (0.08, 0.08, 0.15, 0.15),
        (0.08, 0.15, 0.15, 0.15),
        (0.08, 0.05, 0.15, 0.8),
    ]
    t = 100
    n = 100000
    for index, economy in enumerate(economy_parameters):

        # econ = EvolveEconomy(**economy)
        for sub_index, (natural_rate, rate_shock, original_govt, new_govt) in enumerate(
            example_inputs
        ):
            economy["natural_rate"] = natural_rate
            economy["shock_rate"] = rate_shock
            economy["original_govt"] = original_govt
            economy["long_run_govt"] = new_govt
            econ = EvolveEconomy(**economy)
            sim = econ.simulate(
                t=t,
                n=n,
            )
            plt.figure(
                f"econ_{index}_run_main"
            )  # this needs to aggregate plots across rate scenarios
            plt.plot(
                sim.t,
                sim.y_per_labor,
                label=f"rate shock={rate_shock}, initial govt={original_govt}, end govt={new_govt}",
            )
            plt.figure(f"econ_{index}_run_{sub_index}_labor")
            plt.plot(
                sim.t,
                sim.v1,
                label=f"V1",
            )
            plt.plot(
                sim.t,
                sim.v2,
                label=f"V2",
            )
            plt.plot(
                sim.t,
                sim.v3,
                label=f"V3",
            )
            plt.legend(loc="upper left")
            plt.title(
                f"Labor proportions for rate shock={rate_shock}, initial govt={original_govt}, end govt={new_govt}"
            )
            plt.savefig(f"images/econ_{index}_run_{sub_index}_labor.eps", format="eps")
        plt.figure(f"econ_{index}_run_main")
        plt.title(f"Output per capita")
        plt.legend(loc="upper left")
        plt.savefig(f"images/economy_{index}.eps", format="eps")
        # plt.savefig(f"output_per_labor_{t}.png")
        # plt.clf()
