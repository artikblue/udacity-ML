from sklearn import linear_model


def studentReg(ages_train, net_worths_train):
    reg = None
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    reg = linear_model.LinearRegression()
    reg.fit(ages_train, net_worths_train)
    ### your code goes here!


    return reg
