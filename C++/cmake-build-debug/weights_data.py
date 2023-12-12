f = open("train", "r")
qwe = f.readlines()
ewq = qwe[2]
f.close()
f = open("weights", "w")
print(ewq, file=f)
f.close()
