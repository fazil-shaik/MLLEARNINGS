import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

#create a dataset
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'red_flags_ignored':np.random.randint(0,10,n),
    'talking_stages_failed':np.random.randint(0,16,n),
    'apps_reinstalls':np.random.randint(0,31,n),
    'commitment_phobia':np.round(np.random.uniform(0,10,n),1),
    'stalking_hours_week':np.round(np.random.uniform(0,20,n),1),
    'bio_cringe_level':np.round(np.random.uniform(0,10,n),1),
    'situationship_active':np.random.randint(0,6,n),
    'StandardsVSReality':np.round(np.random.uniform(0,10,n),1),
})


#our goal :- to find out how many years will be we are single

months = np.full(n,2.0)
months+= np.where(
    data['red_flags_ignored'] > 10,
    data['red_flags_ignored'] * 3.0,
    data['red_flags_ignored'] * 0.0
)


months+= np.where(
    data['talking_stages_failed'] > 5,
    data['talking_stages_failed'] * 1.0,
    data['talking_stages_failed'] * 0.4
)

months+= np.where(
    data['apps_reinstalls'] > 15,
    data['apps_reinstalls'] * 0.5,
    data['apps_reinstalls'] * 0.2
)


months+= np.where(
    data['commitment_phobia'] > 15,
    data['commitment_phobia'] * 0.5,
    data['commitment_phobia'] * 0.2
)



months+= np.where(
    data['stalking_hours_week'] > 10,5,
    data['stalking_hours_week'] * 0.4,
)


months+= np.where(
    data['bio_cringe_level'] > 7,8,
    data['bio_cringe_level'] * 1.8,
)


months+= np.where(
    data['situationship_active'] > 2,6,
    data['situationship_active'] * 0.3,
)

months+= np.where(
    data['StandardsVSReality'] > 6,
    data['StandardsVSReality'] * 1.5,
    data['StandardsVSReality'] * 0.3
)

months+= np.random.uniform(-2,2,n)
months = np.clip(np.round(months,1),1,36)


data['months_single'] = months
print("dataset shape",data.shape)
print(data.head)



features = [col for col in data.columns if col!='months_single']

X = data[features]
y = data['months_single']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Training: {X_train.shape[0]} sample  || Testing: {X_test.shape[0]} sample")


#model selection and fitting
dt_model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt_model.fit(X_train,y_train)


#compare linear regression too
model = LinearRegression()
model.fit(X_train,y_train)

print("both models trained!!!")


#evaluate model with  the values and check mse
dt_predict = dt_model.predict(X_test)
model_linear = model.predict(X_test)

dt_mse = mean_squared_error(y_test,dt_predict)
dt_mae = mean_absolute_error(y_test,dt_predict)
r2_dt = r2_score(y_test,dt_predict)

lr_mse = mean_squared_error(y_test,model_linear)
lr_mae = mean_absolute_error(y_test,model_linear)
r2_lr = r2_score(y_test,model_linear)

print("\n"+ "= "* 55)
print("decision vs Linear performace review")
print("\n"+ "= "* 55)
print(f"{'Metrics':<20} {'Decision Tree':<15} {'Linear regression':<15}")

#check and plot actual values
fig,axes = plt.subplot(1,2,figsize = (14,6))
fig.subtitle('Actual vs predicted MOnths single',fontsize = 10,fontweight='bold')
axes[0].scatter(y_test,dt_predict,s=25,alpha=0.5)
axes[0].plot([0,36],[0,36],color='orange',linestyle='-----',linewidth=4)
axes[0].set_Xlabel('Actual')
axes[0].set_Ylabel('Predicted')
axes[0].set_title(f'Decision Tree Predictions {r2_dt:.3f}')
axes[1].scatter(y_test,model_linear,s=25,color='purple')
axes[1].plot([0,36],[0,36],'--',color='red',linewidth=2)
axes[1].set_Xlabel('Actual')
axes[1].set_Ylabel('Predicted')
axes[1].set_title(f'Linear Tree Predictions {r2_lr:.3f}')

plt.tight_layout()
plt.show()


#plot decison tree
fid,ax = plt.subplot(figsize=(20,10))
plot_tree(dt_model,feature_names=features,filled=True,rounded=True,
          ax=ax,max_depth=3,impurity=False)
plt.tight_layout()
plt.show()


#feature importance

# importance = pd.Series(dt_model,feature_importances_,index=features)
