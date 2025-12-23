# # # # data = [10,30,40,12]
# # # # print(data[3])
# # # # print(type(data))


# # # # print(min(data))



# # # def calAnnualrent(data):
# # #     avg = data*12

# # #     return avg

# # # print(calAnnualrent(data=15000))

# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_squared_error,r2_score


# # #workflow for creating of model

# # model = LinearRegression()

# # #data collection

# # X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
# # y = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

# # #train the model

# # model.fit(X,y)
# #goal-to predict rent model in hyderabad
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# #data
# house_size = np.array([500,700,600,1900,2000,4300,1500])
# rent_price = np.array([15000,18000,17000,35000,40000,80000,25000])

# #data in numpy 
# x = house_size.reshape(-1, 1) 
# print(x) # Reshape to 2D array
# y = rent_price

# #create a model
# model = LinearRegression()
# model.fit(x, y)

# #predict rent price
# new_house_size = 500
# predicted_price = model.predict(np.array([[new_house_size]]))
# print(f"The predicted rent price for a house size of {new_house_size} sq ft is: {predicted_price[0]:.0f}")

# #see data
# plt.figure(figsize=(10, 6))
# plt.scatter(house_size, rent_price, color='green', s=100, label='Actual data')  # Fixed 'lable' to 'label'
# plt.plot(house_size, model.predict(x), color='red', linewidth=2, label='Predicted value')  # Fixed 'lable' to 'label'
# plt.scatter(new_house_size, predicted_price, color='yellow', s=200, marker='*', label='Predicted Price')
# plt.xlabel('House sizes in sq.ft', fontsize=14)
# plt.ylabel('Rent prices', fontsize=14)  # Changed to ylabel for clarity
# plt.title('House Size vs Rent Price', fontsize=16)
# plt.savefig('house_rent.png', dpi=300, bbox_inches='tight')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()


