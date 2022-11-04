import numpy as np
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

class LinearSystem:
    def __init__(self, params: DiscreteLinSystemParams, x0: np.ndarray) -> None:
        self.x: np.ndarray= x0
        self.A = params.A
        self.B = params.B
        self.C = params.C
        self.D = params.D

    def update(self, u):
        self.x = self.A @ self.x +self.B @ u

    def getOutput(self, u):
        return self.C * self.x + self.D * u
    
    def getState(self):
        return self.x
    
    def getOrder(self):
        return self.x.size

class Observer:
    def __init__(self, params: DiscreteLinSystemParams, x0) -> None:
        self.current_estimate = x0
        self.A = params.A
        self.B = params.B
        self.C = params.C
        self.D = params.D

    def update(self, observed): pass

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
        observer.update(y)
        x_est = observer.getEstimate()

        evolution[:, i] = x[:, 0]
        estimated[:, i] = x_est[:, 0]
    
    times = np.linspace(0, T, n_step)
    plt.plot(times, evolution[0, :])
    plt.plot(times, estimated[0, :])
    plt.show()

if __name__ == '__main__':
    t_step = 0.01
    x0 = np.array([0, 0]).reshape((-1, 1))
    params = getDiscreteSystemParams(OscillatorParams(10, 1, 1), t_step)
    sys = LinearSystem(params, x0)
    obs = Observer(params, x0)
    mainloop(sys, obs, t_step, T = 10)
