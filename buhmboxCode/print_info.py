import pickle

def print_info():
    with open("info.pickle", "rb") as f:
        arrs, means, stds = pickle.load(f)
    for arr in arrs:
        print(arr)
    for mean in means:
        print(mean)
    for std in stds:
        print(std)

if __name__=="__main__":
    print_info()
