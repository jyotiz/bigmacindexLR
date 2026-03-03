import numpy as np

class LinearRegressionGD1D:
    def __init__(self, alpha=1e-3, n_iters=2000, standardize=True, l2=0.0, verbose=False):
        self.alpha = alpha
        self.n_iters = n_iters
        self.standardize = standardize
        self.l2 = l2
        self.verbose = verbose
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.cost_history = []
        # for inverse-transform if standardize=True
        self._x_mu = None
        self._x_sigma = None
        self._y_mu = None
        self._y_sigma = None

    def _prep_xy(self, x, y):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        assert x.shape[0] == y.shape[0], "x and y must have the same length"

        if self.standardize:
            self._x_mu = x.mean()
            self._x_sigma = x.std() + 1e-8
            xs = (x - self._x_mu) / self._x_sigma

            self._y_mu = y.mean()
            self._y_sigma = y.std() + 1e-8
            ys = (y - self._y_mu) / self._y_sigma
            return xs, ys
        else:
            # still track means for predict (no transform)
            self._x_mu = 0.0
            self._x_sigma = 1.0
            self._y_mu = 0.0
            self._y_sigma = 1.0
            return x, y

    def fit(self, x, y):
        x, y = self._prep_xy(x, y)
        m = x.shape[0]

        # small initialization helps stability
        rng = np.random.default_rng(42)
        self.theta0 = float(rng.normal(0.0, 1e-2))
        self.theta1 = float(rng.normal(0.0, 1e-2))

        prev_cost = np.inf
        for it in range(self.n_iters):
            # predict and residual
            y_pred = self.theta0 + self.theta1 * x
            error = y_pred - y

            # cost (0.5*MSE) + optional L2 (exclude bias)
            cost = 0.5 * np.mean(error**2)
            if self.l2 > 0.0:
                cost += (self.l2 / (2.0)) * (self.theta1**2) / m

            # NaN/Inf guard on cost
            if not np.isfinite(cost):
                if self.verbose:
                    print(f"Non-finite cost at iter {it}. Stopping.")
                break

            self.cost_history.append(cost)

            # gradients (averaged)
            grad0 = np.mean(error)
            grad1 = np.mean(error * x)
            if self.l2 > 0.0:
                grad1 += (self.l2 / m) * self.theta1

            # simple gradient clipping to prevent blow-up
            gnorm = np.hypot(grad0, grad1)
            if gnorm > 1e6:
                scale = 1e6 / (gnorm + 1e-12)
                grad0 *= scale
                grad1 *= scale

            # parameter update
            self.theta0 -= self.alpha * grad0
            self.theta1 -= self.alpha * grad1

            # NaN/Inf guard on params
            if not (np.isfinite(self.theta0) and np.isfinite(self.theta1)):
                if self.verbose:
                    print(f"Non-finite params at iter {it}. Stopping.")
                break

            # simple early stopping on tiny improvement
            if prev_cost - cost < 1e-12:
                if self.verbose:
                    print(f"Early stop at iter {it}, cost={cost:.6e}")
                break
            prev_cost = cost

        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        # use same scaling as training
        xs = (x - self._x_mu) / self._x_sigma
        yhat_scaled = self.theta0 + self.theta1 * xs
        # invert target scaling
        return yhat_scaled * self._y_sigma + self._y_mu