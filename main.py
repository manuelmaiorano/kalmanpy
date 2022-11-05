import numpy as np
import numpy.linalg
import numpy.random
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class OscillatorParams:
    K: float
    M: float
    b: float

@dataclass
class DiscreteLinSystemParams:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

@dataclass
class StatisticalModel:
    Covx: np.ndarray
    Covy: np.ndarray

class LinearSystem:
    def __init__(self, params: DiscreteLinSystemParams, covs: StatisticalModel, x0: np.ndarray) -> None:
        self.x: np.ndarray= x0
        self.A = params.A
        self.B = params.B
        self.C = params.C
        self.D = params.D
        self.Covx = covs.Covx
        self.Covy = covs.Covy
        self.meanx = np.zeros(self.A.shape[0])
        self.meany = np.zeros(self.C.shape[0])

    def update(self, u):
        w =  numpy.random.multivariate_normal(self.meanx, self.Covx)
        self.x = self.A @ self.x +self.B @ u + w

    def getOutput(self, u):
        v =  numpy.random.multivariate_normal(self.meany, self.Covy)
        return self.C @ self.x + self.D * u + v
    
    def getState(self):
        return self.x
    
    def getOrder(self):
        return self.x.size

class Observer:
    def __init__(self, params: DiscreteLinSystemParams, covs: StatisticalModel, x0, cov0) -> None:
        self.current_estimate = x0
        self.current_cov = cov0
        self.A = params.A
        self.B = params.B
        self.C = params.C
        self.D = params.D
        self.Covx = covs.Covx
        self.Covy = covs.Covy

    def update(self, observed, u): 
        x = self.current_estimate
        cov = self.current_cov
        #predict
        x = self.A @ x + self.B @ u
        cov = self.A @ cov @ self.A.T + self.Covx
        #update
        y = observed - self.C @ x
        S = self.C @ cov @ self.C.T + self.Covy
        K = cov @ self.C.T @ numpy.linalg.inv(S)
        x = x + K @ y
        cov = (np.eye(cov.shape[0]) - K @ self.C) @ cov
        
        self.current_estimate = x
        self.cov = cov

    def getEstimate(self):
        return self.current_estimate

def getDiscreteSystemParams(params: OscillatorParams, dt):
    k = params.K
    b = params.b
    m = params.M
    A = np.array([[0, 1],
                  [-k/m , -b/m]])
    A = A*dt + np.eye(A.shape[0])

    B = np.array([0, 1/m]).reshape((-1,1)) * dt
    C = np.array([1, 0]).reshape((1,-1))
    D = 0

    return DiscreteLinSystemParams(A, B, C, D)

def mainloop(system: LinearSystem, observer: Observer, t_step, T):
    n_step = int(T/t_step)
    u = np.array([[1]])

    evolution = np.zeros((system.getOrder(), n_step))
    estimated = np.zeros((system.getOrder(), n_step))
    for i in range(n_step):
        system.update(u)
        x = system.getState()
        y = system.getOutput(u)
        observer.update(y, u)
        x_est = observer.getEstimate()

        evolution[:, i] = x[:, 0]
        estimated[:, i] = x_est[:, 0]
    
    times = np.linspace(0, T, n_step)

    f, (ax1, ax2) = plt.subplots(2, 1) 

    def plot_state(n, ax, title):
        ax.set_title(title)
        ax.plot(times, evolution[n, :])
        ax.plot(times, estimated[n, :])
        ax.legend(['evolution','estimated'])

    plot_state(0, ax1, 'pos')
    plot_state(1, ax2, 'vel')

    plt.show()

if __name__ == '__main__':
    numpy.random.seed(0)

    t_step = 0.01
    x0 = np.array([0, 0]).reshape((-1, 1))
    params = getDiscreteSystemParams(OscillatorParams(10, 1, 1), t_step)
    
    Covx = np.diag([0.5, 0.5])
    Covy = np.diag([10])
    Covs = StatisticalModel(Covx, Covy)

    sys = LinearSystem(params, Covs, x0)
    obs = Observer(params, Covs, x0, Covx)
    mainloop(sys, obs, t_step, T = 10)
