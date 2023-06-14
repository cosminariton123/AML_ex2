import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def fit_small_first(positive_samples, negative_samples):
    #We can get the minimum in O(1)
    a = positive_samples[-1]

    #O(n)
    for elem in negative_samples:
        if elem >= a and elem <= a + 1:
            a = elem - 1 - epsilon
    
    #O(n)
    for elem in positive_samples:
        if elem > a + 1 and elem < a + 2:
            a = elem - 2

    #O(n)
    for elem in negative_samples:
        if elem >= a + 2 and elem <= a + 4:
            a = elem - 4 - epsilon


    return a

def fit_big_first(positive_samples, negative_samples):
    #We can get the maximum in O(1)
    a = positive_samples[-1] - 4

    #O(n)
    for elem in negative_samples:
        if elem >= a + 2 and elem <= a + 4:
            a = elem - 2 + epsilon

    #O(n)
    for elem in positive_samples:
        if elem > a + 1 and elem < a + 2:
            a = elem - 1

    #O(n)    
    for elem in negative_samples:
        if elem >= a and elem <= a + 1:
            a = elem + epsilon


    return a




def A(positive_samples, negative_samples):
    a = -500
    
    #Case 1
    if not positive_samples:
        return a

    #Case 2
    #O(nlogn)
    positive_samples = sorted(positive_samples, reverse=True)
    #O(nlogn)
    negative_samples = sorted(negative_samples, reverse=True)
    a = fit_small_first(positive_samples, negative_samples)
    if loss_function(positive_samples, negative_samples, a) > 0:
        a = fit_big_first(positive_samples[::-1], negative_samples[::-1])

    return a

#O(n)
def loss_function(positive_samples, negative_samples, a_ERM):
    loss = 0

    for elem in negative_samples:
        if elem >= a_ERM and elem <= a_ERM + 1 or elem >= a_ERM + 2 and elem <= a_ERM + 4:
            loss += 1

    for elem in positive_samples:
        if not (elem >= a_ERM and elem <= a_ERM + 1 or elem >= a_ERM + 2 and elem <= a_ERM + 4):
            loss += 1

    return loss

    


def verify(positive_samples, negative_samples, a_ERM):
    for elem in negative_samples:
        if elem >= a_ERM and elem <= a_ERM + 1 or elem >= a_ERM + 2 and elem <= a_ERM + 4:
            print(f"Failure for negative samples. X={elem}, a_ERM={a_ERM}")
            return False

    for elem in positive_samples:
        if not (elem >= a_ERM and elem <= a_ERM + 1 or elem >= a_ERM + 2 and elem <= a_ERM + 4):
            print(f"Failure for positive_samples samples. X={elem}, a_ERM={a_ERM}")
            return False

    return True



def trial(show = False):
    #We are in the realizable case, so we will choose our ground_truth a*
    true_a = np.random.uniform(-10, 10, 1)

    #We will sample n times from a uniform distribution
    S = np.random.uniform(-30, 30, n)

    #O(n)
    positive_samples = [elem for elem in S if elem >= true_a and elem <= true_a + 1 or elem >= true_a + 2 and elem <= true_a + 4]
    #O(n)
    negative_samples = [elem for elem in S if not (elem >= true_a and elem <= true_a + 1 or elem >= true_a + 2 and elem <= true_a + 4) ]

    a_ERM = A(positive_samples, negative_samples)    


    if show is True or not verify(positive_samples, negative_samples, a_ERM):

        plt.scatter(positive_samples, [0 for _ in positive_samples], color="blue", label="pasitive", marker="+")
        plt.scatter(negative_samples, [0 for _ in negative_samples], color="red", label="negative", marker="_")

        plt.vlines([true_a, true_a + 1, true_a + 2, true_a + 4], ymin=-1, ymax=1, colors=["green", "green", "orange", "orange"])
        plt.vlines([a_ERM, a_ERM + 1, a_ERM + 2, a_ERM + 4], ymin=-1, ymax=1, colors=["magenta", "magenta", "black", "black"], linestyles="dashed")

        plt.show()

    return verify(positive_samples, negative_samples, a_ERM)
















#Set this values for higher precision or lower computational time
N = 10**5
n = 60
epsilon = 1e-10

def main():
    #############Complexity analysis#################

    #O(n) - select positives samples from S
    #O(n) - select negative samples from S
    #O(nlogn) - sort positive samples(altough we use n/2 samples, we will consider n for simplicity)
    #O(nlogn) - sort negative samples(same observation as for positives)
    
    #Fit small interval first
    #O(n) - fit first threshold
    #O(n) - fit second threshold
    #O(n) - fit third threshold

    #O(n) - Check if the loss is 0
    
    #Fit big interval first
    #O(n) - fit first threshold
    #O(n) - fit second threshold
    #O(n) - fit third threshold

    #Total
    #O(2 * nlogn + 9n)
    #Which is polynomial => efficient

    n = np.linspace(1, 100, 10**3)
    n_cut_for_exponential = np.linspace(1, 11, 10**3)
    def f(n):
        return 2 * n * np.log(n) + 9 * n
    def exponential(n):
        return 2**n
    def liniar(n):
        return n
    plt.plot(n, f(n), label="O(2 * nlogn + 9n)")
    plt.plot(n_cut_for_exponential, exponential(n_cut_for_exponential), label="O(2**n) for reference")
    plt.plot(n, liniar(n), label="O(n) for reference")
    plt.title("Close this plot to continue to probabilistic testing")
    plt.legend()
    plt.show()
    
    #######################Probabilistic testing###############################
    #The testing will halt if a fail is detected and a graphic representation will be shown
    results = list(map(trial,  tqdm([False for _ in range(N)], desc="Started probabilistic testing")))    

    if np.array(results).all() != True:
        print("Fail")
    else:
        print("Succes")

if __name__ == "__main__":
    main()
