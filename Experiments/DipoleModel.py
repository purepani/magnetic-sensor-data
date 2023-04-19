import scipy as sp
import numpy as np



class Dipole:
    def __init__(self, M0, diameter, length):
        self.M0 = M0
        self.diameter = diameter
        self.length=length

    def B_dipole(self, position, rotation):
        M0 = self.M0
        shape=(self.diameter, self.length)
        R = np.repeat(np.sqrt(np.sum(position**2, axis=1))[:, np.newaxis], 3, axis=1)
        BT = M0*(shape[0]/2)**2 *shape[1]*np.pi/(4*np.pi)
        B = BT*((3*position*(np.matmul(position, rotation)[:, np.newaxis]))/R**5 - rotation/R**3)
        return B

    def getField_dipole(self, x, positions):
        position=x[:3]
        axis=x[3:]
        #phi = x[3]
        #theta = x[4]
        #axis = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        return self.B_dipole(positions-position, axis)

    def cost_dipole(self, x, B, positions):
        diff = self.getField_dipole(x, positions)-B
        return np.sum(diff**2)

    def __call__(self, B, positions, bounds =([-300, -300, 0, -1, -1, 0], [300, 300, 300, 1, 1, 1]), guess=np.array([0,0,0,0, 0, 1])):
        bounds = zip(*bounds)
        cons = [{'type':'eq', 'fun': lambda x: np.sum(x[3:]**2)-1}]
        sol = sp.optimize.minimize(self.cost_dipole, x0=guess, args=(B, positions), bounds=bounds, tol=1e-2000, constraints = cons)
        return sol.x



