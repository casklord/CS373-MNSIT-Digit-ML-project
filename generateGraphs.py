import numpy as np
from matplotlib import pyplot
import classifier


X_train, X_test, y_train, y_test = classifier.getdata()



# accuracy vs hyperparameter graphs

#comparing number of epochs with overall accuracy


resEp100 = classifier.run(X_train, X_test, y_train, y_test, 100, 0.01, 0.5)


y = []
x = []
for i, l in enumerate(resEp100):
    y.append(np.mean(resEp100[i]))
    x.append(i+1)

pyplot.plot(x, y)
pyplot.title('Number of Epochs vs Overall Accuracy')
pyplot.xlabel('Number of Epochs')
pyplot.ylabel('Accuracy')
pyplot.ylim(0.94, 0.98)
pyplot.show()


#comparing overall accuracy with different penalty  



y = []
x = []
penalties = [0.1, 0.2, 0.5, 1, 2, 5]
for p in penalties:

    y.append(np.mean(classifier.run(X_train, X_test, y_train, y_test, 1, 0.01, p)))
    x.append(p)



pyplot.plot(x, y)
pyplot.title('Penalty vs Overall Accuracy')
pyplot.xlabel('Penalty')
pyplot.ylabel('Accuracy')
pyplot.show()

#comparing overall accuracy with different gamma



y = []
x = []
gammas = [0.0001, 0.001, 0.01, 0.1, 0.5]
for g in gammas:

    y.append(np.mean(classifier.run(X_train, X_test, y_train, y_test, 1, g, 0.5)))
    x.append(g)

pyplot.plot(x, y)
pyplot.title('Gamma vs Overall Accuracy')
pyplot.xlabel('Gamma')
pyplot.ylabel('Accuracy')
print (y)
pyplot.show()


#compare which digits train more accurately


resCompDigits = classifier.run(X_train, X_test, y_train, y_test, 1, 0.01, 0.5)
y = resCompDigits[0]
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
pyplot.bar(x, y)
pyplot.ylim(0.85, 1)
pyplot.title('Digit vs Training Accuracy')
pyplot.xlabel('Digit')
pyplot.ylabel('Accuracy')
pyplot.show()



