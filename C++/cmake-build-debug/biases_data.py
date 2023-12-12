f = open("train", "r")
qwe = f.readlines()
ewq = qwe[3]
f.close()
f = open("biases", "w")
print(ewq, file=f)
f.close()
