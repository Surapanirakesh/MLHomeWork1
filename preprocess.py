import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


traindata = pd.read_csv("datasets/train.csv")
testdata = pd.read_csv("datasets/test.csv")
combinedata = [traindata,testdata]
print(combinedata)
print(" training data set info")
print(traindata.info())
print("Missings in the features in training data set")
print(traindata.isna().sum())
print("Missings in the features in testing data set")
print(testdata.isna().sum())
print("numerical parameters properties")
print(traindata.describe())

categoricalparameter = ["Survived","Pclass","Sex","Embarked"]

print("count of each categorical parameter")
print(traindata[categoricalparameter].count())
print("count of each categorical parameter")

for x in categoricalparameter:
    print(" Unique values in " + x)
    print(traindata[x].unique())
    print(" Most frequent values in " + x)
    print(traindata[x].mode().values)
    print(" frequent is " + x)
    print(traindata[x].value_counts().max())

print("correlation for Pclass = 1 and survived")
Pclass1Survived = traindata[(traindata.Pclass == 1)&(traindata.Survived == 1)]
Pclass1data = traindata[(traindata.Pclass == 1)]
print(Pclass1Survived["Pclass"].count()/Pclass1data["Pclass"].count())



femalesurvival = traindata[(traindata.Sex == "female") & (traindata.Survived == 1)]
totalsurvival = traindata[(traindata.Survived == 1)]
femaleSurvivalPercentage = (femalesurvival["Survived"].count() / totalsurvival["Survived"].count()) * 100
print(" Female survival percentage ")
print(femaleSurvivalPercentage)

agesurvivaldata = traindata[(traindata.Survived == 1)]
# agesurvivaldataset = agesurvivaldata["Age"].value_counts().keys()
ageorder = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80]
# print(sorted(agesurvivaldataset))
# print(agesurvivaldata["Age"].value_counts().max())
plt.hist(agesurvivaldata["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 1 | total = "+str(agesurvivaldata["Age"].count())) #TOTAL IS NOT NEEDED
plt.grid(axis= "x", alpha = 0.75)
plt.grid(axis= "y", alpha = 0.75)
plt.show()

agedeathdata = traindata[(traindata.Survived == 0)]
# agedeathdata["Age"] = agedeathdata["Age"].fillna(agedeathdata["Age"].mode())
agedeathdataset = agedeathdata["Age"].value_counts().keys()
print(agedeathdata["Age"].value_counts())
plt.hist(agedeathdata["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 0 | total = "+ str(agesurvivaldata["Age"].count()))
plt.show()


pclass1survival = traindata[(traindata.Survived == 1) & (traindata.Pclass == 1)]
pclass1death = traindata[(traindata.Survived == 0) & (traindata.Pclass == 1)]
pclass2survival = traindata[(traindata.Survived == 1) & (traindata.Pclass == 2)]
pclass2death = traindata[(traindata.Survived == 0) & (traindata.Pclass == 2)]
pclass3survival = traindata[(traindata.Survived == 1) & (traindata.Pclass == 3)]
pclass3death = traindata[(traindata.Survived == 0) & (traindata.Pclass == 3)]

plt.hist(pclass1survival["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 1 | Pclass = 1 | total = "+ str(pclass1survival["Age"].count()))
plt.show()

plt.hist(pclass1death["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 0 | Pclass = 1 | total = "+ str(pclass1death["Age"].count()))
plt.show()

plt.hist(pclass2survival["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 1 | Pclass = 2 | total = "+ str(pclass2survival["Age"].count()))
plt.show()

plt.hist(pclass2death["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 0 | Pclass = 2 | total = "+ str(pclass2death["Age"].count()))
plt.show()

plt.hist(pclass3survival["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 1 | Pclass = 3 | total = "+ str(pclass3survival["Age"].count()))
plt.show()

plt.hist(pclass3death["Age"], ageorder)
plt.xlabel("Age")
plt.title("Survived = 0 | Pclass = 3 | total = "+ str(pclass3death["Age"].count()))
plt.show()

EmbarkedSsurvival = traindata[(traindata.Survived == 1) & (traindata.Embarked == "S") ]

AvgSsurvival = [EmbarkedSsurvival[(traindata.Sex == "female")]["Fare"].mean(),EmbarkedSsurvival[(traindata.Sex == "male")]["Fare"].mean()]
x = ["female", "male"]
EmbarkedSdeath = traindata[(traindata.Survived == 0) & (traindata.Embarked == "S")]
AvgSdeath = [EmbarkedSdeath[(traindata.Sex == "female")]["Fare"].mean(),EmbarkedSdeath[(traindata.Sex == "male")]["Fare"].mean()]
EmbarkedCsurvival = traindata[(traindata.Survived == 1) & (traindata.Embarked == "C")]
AvgCsurvival = [EmbarkedCsurvival[(traindata.Sex == "female")]["Fare"].mean(),EmbarkedCsurvival[(traindata.Sex == "male")]["Fare"].mean()]
EmbarkedCdeath = traindata[(traindata.Survived == 0) & (traindata.Embarked == "C")]
AvgCdeath = [EmbarkedCdeath[(traindata.Sex == "female")]["Fare"].mean(),EmbarkedCdeath[(traindata.Sex == "male")]["Fare"].mean()]
EmbarkedQsurvival = traindata[(traindata.Survived == 1) & (traindata.Embarked == "Q")]
AvgQsurvival = [EmbarkedQsurvival[(traindata.Sex == "female")]["Fare"].mean(),EmbarkedQsurvival[(traindata.Sex == "male")]["Fare"].mean()]
EmbarkedQdeath = traindata[(traindata.Survived == 0) & (traindata.Embarked == "Q")]
AvgQdeath = [EmbarkedQdeath[(traindata.Sex == "female")]["Fare"].mean(),EmbarkedQdeath[(traindata.Sex == "male")]["Fare"].mean()]
yheight = np.arange(len(x))
plt.bar( yheight , AvgSsurvival,align= "center", alpha = 0.5)
plt.xticks(yheight, x)
plt.xlabel("Gender")
plt.ylabel("Fare")
plt.title("Survived = 1 | Embarked = S | total = "+ str(EmbarkedSsurvival["Sex"].count()))
plt.show()

plt.bar( yheight , AvgSdeath,align= "center", alpha = 0.5)
plt.xticks(yheight, x)
plt.xlabel("Gender")
plt.ylabel("Fare")
plt.title("Survived = 0 | Embarked = S | total = "+ str(EmbarkedSdeath["Sex"].count()))
plt.show()

plt.bar( yheight , AvgCsurvival,align= "center", alpha = 0.5)
plt.xticks(yheight, x)
plt.xlabel("Gender")
plt.ylabel("Fare")
plt.title("Survived = 1 | Embarked = C | total = "+ str(EmbarkedCsurvival["Sex"].count()))
plt.show()

plt.bar( yheight , AvgCdeath,align= "center", alpha = 0.5)
plt.xticks(yheight, x)
plt.xlabel("Gender")
plt.ylabel("Fare")
plt.title("Survived = 0 | Embarked = C | total = "+ str(EmbarkedCdeath["Sex"].count()))
plt.show()

plt.bar( yheight , AvgQsurvival,align= "center", alpha = 0.5)
plt.xticks(yheight, x)
plt.xlabel("Gender")
plt.ylabel("Fare")
plt.title("Survived = 1 | Embarked = Q | total = "+ str(EmbarkedQsurvival["Sex"].count()))
plt.show()

plt.bar( yheight , AvgQdeath,align= "center", alpha = 0.5)
plt.xticks(yheight, x)
plt.xlabel("Gender")
plt.ylabel("Fare")
plt.title("Survived = 0 | Embarked = Q | total = "+ str(EmbarkedQdeath["Sex"].count()))
plt.show()


rateofduplication = 1 - ( len(traindata["Ticket"].drop_duplicates()) / traindata["Ticket"].count())
print("Rate of Duplication in Tickets")
print(rateofduplication)
print("dropping the Ticket Feature")
traindata = traindata.drop(['Ticket'],axis=1)
testdata = testdata.drop(['Ticket'],axis=1)

print("null values count for cabin feature in both training and testing datasets")
print(testdata["Cabin"].isna().sum() + traindata["Cabin"].isna().sum())
traindata = traindata.drop(['Cabin'],axis=1)
testdata = testdata.drop(['Cabin'],axis=1)
combinedata = [traindata,testdata]
print(combinedata)

print("filling Age NA values with random values ")
print(traindata.isna().sum())
print(testdata.isna().sum())
def missingvalue(data):
    if np.isnan(data)== True:
        data = np.random.normal(mean,std,1)

    return data
for dataset in combinedata:
    mean = dataset["Age"].mean()
    std = dataset["Age"].std()
    dataset['Age'] = dataset['Age'].apply(missingvalue)

traindata = combinedata[0]
testdata = combinedata[1]
print(traindata.isna().sum())
print(testdata.isna().sum())

Genderconvertion ={'male':0, 'female':1}
for dataset in combinedata:
    Gender = pd.DataFrame([Genderconvertion[item] for item in dataset["Sex"]],columns= ['Gender'])
    dataset = dataset.append(Gender)
    dataset = dataset.drop(['Sex'], axis=1)
    print("convert Sex feature to a new feature called Gender")
    print(dataset.info())

traindata = combinedata[0]
testdata = combinedata[1]


print("filling Embarked NA values with Mode in train data")
print(traindata['Embarked'].isna().sum())
modevalue = traindata['Embarked'].mode().values
traindata['Embarked'] = traindata['Embarked'].fillna(modevalue[0])
print(traindata['Embarked'].isna().sum())

print("filling Fare NA values with Mode in test data")
print(testdata['Fare'].isna().sum())
modevalue = testdata['Fare'].mode().values
testdata['Fare'] = testdata['Fare'].fillna(modevalue[0])
print(testdata['Fare'].isna().sum())


for dataset in combinedata:

  dataset.loc[ (dataset['Fare'] <= 7.91 ), 'Fare'] = 0
  dataset.loc[((dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454)), 'Fare'] = 1
  dataset.loc[((dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0)), 'Fare'] = 2
  dataset.loc[dataset['Fare'] > 31.0, 'Fare'] = 3
  print("Fare updated as per FareBand")
  print((dataset['Fare']))


traindata = combinedata[0]
testdata = combinedata[1]

survivedfare0 = traindata[(traindata.Fare == 0) & (traindata.Survived == 1) ]
totalfare0 = traindata[(traindata.Fare == 0)]

print("Fare <= 7.91 Survived")
print(survivedfare0["Fare"].count()/totalfare0["Fare"].count())

survivedfare1 = traindata[(traindata.Fare == 1) & (traindata.Survived == 1) ]
totalfare1 = traindata[(traindata.Fare == 1)]

print("Fare> 7.91 & Fare <= 14.454 Survived")
print(survivedfare1["Fare"].count()/totalfare1["Fare"].count())

survivedfare2 = traindata[(traindata.Fare == 2) & (traindata.Survived == 1) ]
totalfare2 = traindata[(traindata.Fare == 2)]

print("Fare<= 31.0 & Fare >14.454 Survived")
print(survivedfare2["Fare"].count()/totalfare2["Fare"].count())

survivedfare3 = traindata[(traindata.Fare == 3) & (traindata.Survived == 1) ]
totalfare3 = traindata[(traindata.Fare == 3)]

print("Fare >31.0 Survived")
print(survivedfare3["Fare"].count()/totalfare3["Fare"].count())
