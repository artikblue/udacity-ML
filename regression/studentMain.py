#!/usr/bin/python

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from studentRegression import studentReg
from class_vis import prettyPicture, output_image
from sklearn.metrics import mean_absolute_error
from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



reg = studentReg(ages_train, net_worths_train)

y_pred = reg.predict(ages_test)
print(y_pred)
print(net_worths_test)

print(reg.coef_)
print(reg.intercept_)
print(reg.score(ages_test, net_worths_test))
print(reg.score(ages_train, net_worths_train))

error = mean_absolute_error(y_pred, net_worths_test)
print(error)

plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())
