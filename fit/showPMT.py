import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


r=21/40
pmt_mid=np.array([[1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), 1/np.sqrt(2)],
                [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), 1/np.sqrt(2)],
                [0,1,0],
                [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                [-1,0,0],
                [-1/np.sqrt(2), -1/np.sqrt(2),0],
                [1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), -1/np.sqrt(2)],
                [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), 1/np.sqrt(2)],
                [1/np.sqrt(2), 0,1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), 1/np.sqrt(2)],
                [0, -1, 0],
                [1/np.sqrt(2), -1/np.sqrt(2), 0],
                [1, 0, 0],
                [1/np.sqrt(2), 1/np.sqrt(2), 0],
                [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), -1/np.sqrt(2)],
                [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), -1/np.sqrt(2)],
                ])


pmt_r=np.zeros((20,3))
pmt_up=np.zeros((20,3))

for i, pmt in enumerate(range(20)):
    a=pmt_mid[pmt][0]
    b=pmt_mid[pmt][1]
    pmt_r[i]=[-0.5*r*b/np.sqrt(a**2+b**2), 0.5*r*a/np.sqrt(a**2+b**2), 0]
    pmt_up[i]=np.cross(pmt_mid[pmt], pmt_r[i])


while True:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-1.5,1.5)


    for i in range(20):
        if np.isin(i, [0,1,2,4,6,7,8,10,13,15,17,18]):
            alpha=1
        else:
            alpha=0
        ax.quiver(pmt_mid[i,0]+pmt_up[i,0], pmt_mid[i,1]+pmt_up[i,1], pmt_mid[i,2]+pmt_up[i,2], pmt_r[i,0], pmt_r[i,1], pmt_r[i,2], linewidth=5, color='k', alpha=alpha)
        ax.quiver(pmt_mid[i,0]+pmt_up[i,0], pmt_mid[i,1]+pmt_up[i,1], pmt_mid[i,2]+pmt_up[i,2], -pmt_r[i,0], -pmt_r[i,1], -pmt_r[i,2], linewidth=5, color='k', alpha=alpha)
        ax.quiver(pmt_mid[i,0]-pmt_up[i,0], pmt_mid[i,1]-pmt_up[i,1], pmt_mid[i,2]-pmt_up[i,2], pmt_r[i,0], pmt_r[i,1], pmt_r[i,2], linewidth=5, color='k', alpha=alpha)
        ax.quiver(pmt_mid[i,0]-pmt_up[i,0], pmt_mid[i,1]-pmt_up[i,1], pmt_mid[i,2]-pmt_up[i,2], -pmt_r[i,0], -pmt_r[i,1], -pmt_r[i,2], linewidth=5, color='k', alpha=alpha)

        ax.quiver(pmt_mid[i,0]+pmt_r[i,0], pmt_mid[i,1]+pmt_r[i,1], pmt_mid[i,2]+pmt_r[i,2], pmt_up[i,0], pmt_up[i,1], pmt_up[i,2], linewidth=5, color='k', alpha=alpha)
        ax.quiver(pmt_mid[i,0]+pmt_r[i,0], pmt_mid[i,1]+pmt_r[i,1], pmt_mid[i,2]+pmt_r[i,2], -pmt_up[i,0], -pmt_up[i,1], -pmt_up[i,2], linewidth=5, color='k', alpha=alpha)
        ax.quiver(pmt_mid[i,0]-pmt_r[i,0], pmt_mid[i,1]-pmt_r[i,1], pmt_mid[i,2]-pmt_r[i,2], pmt_up[i,0], pmt_up[i,1], pmt_up[i,2], linewidth=5, color='k', alpha=alpha)
        ax.quiver(pmt_mid[i,0]-pmt_r[i,0], pmt_mid[i,1]-pmt_r[i,1], pmt_mid[i,2]-pmt_r[i,2], -pmt_up[i,0], -pmt_up[i,1], -pmt_up[i,2], linewidth=5, color='k', alpha=alpha)



    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 0.25 * np.outer(np.cos(u), np.sin(v))
    y = 0.25 * np.outer(np.sin(u), np.sin(v))
    z = 0.25 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    x = 0.75 * np.outer(np.cos(u), np.sin(v))
    y = 0.75 * np.outer(np.sin(u), np.sin(v))
    z = 0.75 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.2)

    # x=-2
    # phi=np.random.uniform(0, 2*np.pi, 10)
    # r2=np.random.uniform(0,0.25**2, 10)
    # y=np.sqrt(r2)*np.cos(phi)
    # z=np.sqrt(r2)*np.sin(phi)
    #
    # for i in range(len(y)):
    #     ax.quiver(x,y,z,0.5,0,0, color='r')
    #
    #
    # y=-2
    # phi=np.random.uniform(0, 2*np.pi, 10)
    # r2=np.random.uniform(0,0.25**2, 10)
    # x=np.sqrt(r2)*np.cos(phi)
    # z=np.sqrt(r2)*np.sin(phi)
    #
    # for i in range(len(x)):
    #     ax.quiver(x,y,z,0,0.5,0, color='r')
    #
    # costheta=np.random.uniform(-1,1)
    # phi=np.random.uniform(0,2*np.pi)
    # r3=np.random.uniform(0,(10/40)**3)
    # r=r3**(1/3)
    # v=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
    #
    # costheta=np.random.uniform(-1,1, 25)
    # phi=np.random.uniform(0,2*np.pi, 25)
    # us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
    # us=us.T
    #
    # for i in range(len(phi)):
    #     ax.quiver(v[0], v[1], v[2], 0.5*us[i,0], 0.5*us[i,1], 0.5*us[i,2], color='y')

    plt.show()
