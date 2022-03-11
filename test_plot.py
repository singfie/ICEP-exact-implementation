from matplotlib import pyplot as plt
from scipy.stats import skewnorm

def main():
    test = skewnorm.rvs(a=-2, loc=3000, scale=0.3*3000,size=10000)
    plt.hist(test, 100)
    plt.show()

if __name__=="__main__":
    main()