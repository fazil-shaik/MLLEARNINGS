import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data

commonIntrest = [2,3,4,5,6,8,9,7,6,7]
response_time = [3,4,5,2,1,5,8,9,8,3]
age_compatibility = [4,5,7,8,9,9,8,9,7,6]

#wanted to predit match score of each couple out of 100
match_score = [25,35,50,55,65,70,75,80,90,100]

#prepare the model

X = np.array([commonIntrest,response_time,age_compatibility]).T
Y = np.array(match_score)
print("Our features {X}:")
print(f"Our features {X}:")
print(X)

print(f"\nShape:{X.shape}")


#create a load model
model = LinearRegression()
model.fit(X,Y)

print(f"\n -----what model had learned----!")
print(f"coefficients: {model.coef_.round(2)}")
print(f"----->coefficnets of common_intrest: {model.coef_[0]:.2f} points per unit")
print(f"----->coefficnets of response_time: {model.coef_[1]:.2f} points per unit")
print(f"----->coefficnets of age_compatibilty: {model.coef_[2]:.2f} points per unit")

#predict new match
# y=w1x1+w2x2+w3x3+b

new_person = [[7,8,6]]
new_person = [[7,8,6]]
predict_score = model.predict(new_person)
#print predictions

print(f"\n---------new match predictions-----------")
print(f"Comman intrests 7/10")
print(f"Response time 8/10")
print(f"Age compatibiltiy 6/10")

print(f"Predicted Match score {predict_score[0]:.1f}")

#visualize

fig,axis=plt.subplots(1,3,figsize=(14,5))
features = [commonIntrest,response_time,age_compatibility]
names=['common Intest','response Time','age compatibility']
colors=["#008080", "#7AE07A", "#0000FF"]


for i,(feature,name,color) in enumerate(zip(features,names,colors)):
    axis[i].scatter(feature,match_score,color=color,s=100,alpha=0.7)
    axis[i].set_ylabel("Match Score",fontsize=11)
    axis[i].set_title(f"{names} vs Match Score",fontsize=14)
    axis[i].set_title(f"{name} vs Match Score",fontsize=14)

# plt.subtitle("Match Compatibility Analysis",fontsize=16,alpha=0.8,bold=True)
plt.suptitle("Match Compatibility Analysis",fontsize=16,alpha=0.8,weight='bold')
plt.savefig("MatchscoreAnalysis.png",dpi=200, bbox_inches="tight")
plt.show()


print(f"\n-----End of the Analysis-----")



