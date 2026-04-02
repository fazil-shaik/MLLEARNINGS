# from sklearn.metrics import classification_report
# import numpy as np
# import pandas as pd
# from sklearn.metrics import r2_score,mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVR
# import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import StandardScaler


# np.random.seed(42)
# n = 1000

# mass = np.random.uniform(1e10, 1e26, n)
# radius = np.random.uniform(1e5, 1e8, n)
# density = mass / ((4/3) * np.pi * radius**3)
# surface_area = 4 * np.pi * radius**2
# noise = np.random.normal(1.0, 0.02, n)

# G = 6.674e-11
# gravity = (G * mass / radius**2) * noise

# df = pd.DataFrame({
#     'mass_kg': mass,
#     'radius_m': radius,
#     'density_kg_m3': density,
#     'surface_area_m2': surface_area,
#     'surface_gravity_m_s2': gravity
# })

# df['log_mass']    = np.log(df['mass_kg'])
# df['log_radius']  = np.log(df['radius_m'])
# df['log_density'] = np.log(df['density_kg_m3'])
# df['log_area']    = np.log(df['surface_area_m2'])
# df['log_gravity'] = np.log(df['surface_gravity_m_s2'])

# X = df[['log_mass', 'log_radius', 'log_density', 'log_area']]
# y = df['log_gravity']   
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_log_pred = model.predict(X_test)

# # print("predictions are "+str(y_log_pred))


# SupportModel = SVR(kernel='rbf',C=100,gamma=0.1,epsilon=0.1)
# SupportModel.fit(X_train,y_train)


# r2 = r2_score(y_test,y_log_pred)
# print(f'R^2 Score: {r2:.2f}')


# plt.style.use('dark_background')
# fig,axes = plt.subplots(1,2,figsize=(12,5))
# axes[0].scatter(y_test,y_log_pred,alpha=0.5,color='blue')
# axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# axes[0].set_xlabel('Actual Gravity (m/s²)')
# axes[0].set_ylabel('Predicted Gravity (m/s²)')
# axes[0].set_title('Actual vs Predicted Gravity')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
# residuals = y_test-y_log_pred
# axes[1].scatter(y_test, y_log_pred, alpha=0.5, color='magenta')
# axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')   
# axes[1].set_xlabel('Actual log(Gravity)')
# axes[1].set_ylabel('Predicted log(Gravity)')
# axes[1].set_title('Polynomial Model: Actual vs Predicted log(Gravity)')
# axes[1].grid(True, which="both", ls="--", linewidth=0.5)
# axes[1].set_aspect('equal', adjustable='box')
# axes[1].legend(['Perfect Prediction', 'Predicted vs Actual'], loc='upper left')
# fig, ax = plt.subplots(figsize=(8, 5))
# fig.patch.set_facecolor('#0f0f0f')
# ax.set_facecolor('#1a1a1a')

# ax.scatter(y_log_pred, residuals,
#            alpha=0.4, color='darkorange',
#            edgecolors='white', linewidths=0.3, s=40)

# ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Zero error line')

# ax.set_xlabel('Predicted log(gravity)', color='white', fontsize=12)
# ax.set_ylabel('Residual (Actual - Predicted)', color='white', fontsize=12)
# ax.set_title('Residual Plot\n(Random scatter around 0 = good fit)', color='white', fontsize=13)
# ax.tick_params(colors='white')
# ax.legend(labelcolor='white', facecolor='#2a2a2a')
# ax.grid(True, alpha=0.2)

# for spine in ax.spines.values():
#     spine.set_edgecolor('#444')

# plt.tight_layout()
# plt.show()
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)


# train_losses = []
# test_losses = []

# sgd = SGDRegressor(max_iter=100, warm_start=True,    # warm_start keeps weights between calls
#                    learning_rate='constant',
#                    eta0=0.001, random_state=42)

# for epoch in range(200):
#     sgd.fit(X_train_scaled, y_train)

#     # MSE on train and test each epoch
#     train_pred = sgd.predict(X_train_scaled)
#     test_pred  = sgd.predict(X_test_scaled)

#     train_mse  = np.mean((y_train.values - train_pred) ** 2)
#     test_mse   = np.mean((y_test.values  - test_pred)  ** 2)

#     train_losses.append(train_mse)
#     test_losses.append(test_mse)




# # fig, ax = plt.subplots(figsize=(9, 5))
# # fig.patch.set_facecolor('#0f0f0f')
# # ax.set_facecolor('#1a1a1a')

# # ax.plot(train_losses, color='#4fc3f7', linewidth=2, label='Train loss')
# # ax.plot(test_losses,  color='#ff8a65', linewidth=2,
# #         linestyle='--', label='Test loss')

# # ax.set_xlabel('Epoch', color='white', fontsize=12)
# # ax.set_ylabel('MSE Loss', color='white', fontsize=12)
# # ax.set_title('Loss Curve\n(Both falling = learning well)', color='white', fontsize=13)
# # ax.tick_params(colors='white')
# # ax.legend(labelcolor='white', facecolor='#2a2a2a')
# # ax.grid(True, alpha=0.2)

# # for spine in ax.spines.values():
# #     spine.set_edgecolor('#444')

# # plt.tight_layout()
# # plt.show()

# # y_pred_actual = np.exp(y_log_pred)
# # y_test_actual = np.exp(y_test)

# # print("=== Log-space R² (how well model fits log relationship) ===")
# # print(f"R² (log space) : {r2_score(y_test, y_log_pred):.6f}")

# # print("\n=== Actual gravity space R² ===")
# # print(f"R² (actual)    : {r2_score(y_test_actual, y_pred_actual):.6f}")

# # print("\n=== What the model learned ===")
# # for feat, coef in zip(X.columns, model.coef_):
# #     print(f"  {feat:15s} → coefficient: {coef:.4f}")
# # print(f"  {'intercept':15s} → {model.intercept_:.4f}")

# # print("\n=== Physics check ===")
# # print("Expected: log_mass coef ≈ +1.0, log_radius coef ≈ -2.0")
# # print("(because g = G·M·r⁻²  →  log(g) = log(G) + 1·log(M) - 2·log(r))")



# #decision Tree regression

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.tree import plot_tree

# np.random.seed(42)

# n = 300

# X1 = np.random.rand(n) * 10
# X2 = np.random.rand(n) * 5
# X3 = np.random.rand(n) * 7
# X_noise1 = np.random.randn(n)
# X_noise2 = np.random.randn(n)
# X_noise3 = np.random.randn(n)
# X = np.column_stack([X1, X2, X3,X_noise1, X_noise2,X_noise3])

# y = 5000*X1 + 10000*(X2**2) - 1500*(X3**3) + np.random.randn(n)*10000

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# feature_names = ["Size", "Rooms", "distance","Noise1", "Noise2","Noise3"]

# dt = DecisionTreeRegressor(max_depth=3,random_state=42)
# dt.fit(X_train,y_train)

# importance = dt.feature_importances_
# print(importance)
# plt.style.use('dark_background')

# plot_tree(
#     dt,feature_names=feature_names,rounded=True,max_depth=2
# )

# plt.show()

# plt.style.use('dark_background')
# plt.figure()
# plt.bar(feature_names, importance)
# plt.xlabel("Features")
# plt.ylabel("Importance Score")
# plt.title("Feature Importance - Decision Tree")
# plt.show()



# #histogram of feature importance
# # indices = np.argsort(importance)[::-1]

# plt.style.use('dark_background')
# plt.figure(figsize=(10,5))
# plt.bar(feature_names, importance, color='#4fc3f7')
# plt.xlabel("Features", color='white')
# plt.ylabel("Importance Score", color='white')
# plt.title("Feature Importance - Decision Tree", color='white')
# plt.xticks(rotation=45, color='white')
# plt.yticks(color='white')
# plt.grid(True, alpha=0.2)
# plt.tight_layout()
# plt.show()




#logistic regression


from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
size = 300

dataValues = {
    'monthly_income_inr':np.random.randint(10000,30000,size=size),
    'income_volatility':np.random.randint(2,12,size=size),
    'loom_age_years':np.random.randint(10,20,size=size),
    'children_in_school':np.random.randint(2,8,size=size),
    'distance_to_city_km':np.random.randint(5,15,size=size),
    'spouse_employed':np.random.uniform(0,1,size=size),
    'will_abandon':np.random.uniform(0,1,size=size)
}


income = dataValues['monthly_income_inr']
votality = dataValues['income_volatility']
loom_age=dataValues['loom_age_years']
children = dataValues['children_in_school']
distance = dataValues['distance_to_city_km']
spouse = dataValues['spouse_employed']

income_scaled = (income - income.mean()) / income.std()

z = (
    -0.0005*income
    +0.12*votality
    + 0.04   * loom_age       # older loom → more frustration
    + 0.18   * children       # more kids in school → financial pressure
    + 0.06   * distance      # closer to city → migration pull stronger
    - 1.5    * spouse         # spouse earning → weaver feels safer staying
    + 0.5 
)

P = 1 / (1 + np.exp(-z))

dataValues['will_abandon'] = np.random.binomial(1, P)


print(P.min().round(2), P.max().round(2))


df = pd.DataFrame(dataValues)


X = df.drop(['will_abandon'],axis=1)
y = df['will_abandon']

print(P.min().round(2), P.max().round(2))
print(pd.Series(dataValues['will_abandon']).value_counts())

print('sizes are ',X.shape,y.shape)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_predict = model.predict(X)


plt.style.use('dark_background')
# Plot
plt.figure(figsize=(12, 6))
plt.scatter(X['monthly_income_inr'], y, color='red', label='Actual data')
plt.scatter(X['monthly_income_inr'], P, color='blue', label='Predicted probability')
plt.xlabel("Monthly Income (INR)")
plt.ylabel("Will Abandon (1=Yes, 0=No)")
plt.legend()
plt.title("Logistic Regression: Probability of Abandonment vs Monthly Income")
plt.grid(True, alpha=0.2)
plt.show()

#actual vs predicted
plt.figure(figsize=(12, 6))
plt.scatter(y, y_predict, color='purple', label='Actual vs Predicted')
plt.xlabel("Actual Will Abandon (1=Yes, 0=No)")
plt.ylabel("Predicted Will Abandon (1=Yes, 0=No)")
plt.title("Logistic Regression: Actual vs Predicted")
plt.legend()    
plt.grid(True, alpha=0.2)
plt.show()

#features importance
fig,axes = plt.subplots(1,1,figsize=(10,5))
axes.bar(X.columns, model.coef_[0], color='#4fc3f7')
axes.set_xlabel("Features", color='white')
axes.set_ylabel("Coefficient Value", color='white') 
axes.set_title("Feature Coefficients - Logistic Regression", color='white')
axes.tick_params(colors='white')
axes.grid(True, alpha=0.2)
plt.show()

# print("=== What the model learned ===")
for feat, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feat:20s} → coefficient: {coef:.4f}")
print(f"  {'intercept':20s} → {model.intercept_[0]:.4f}")