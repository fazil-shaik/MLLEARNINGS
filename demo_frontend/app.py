# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt

# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import r2_score, mean_absolute_error


# # st.title("Linear Regression Demo")

# # st.write("Predict Y from X using Linear Regression")


# # # Sample dataset

# # data = pd.DataFrame({
# #     "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
# #     "Marks": [35, 40, 45, 50, 55, 60, 65, 72, 78, 85]
# # })

# # st.subheader("Dataset")

# # st.dataframe(data)


# # # Prepare X and y

# # X = data[["Hours"]]
# # y = data["Marks"]


# # # Train model

# # model = LinearRegression()

# # model.fit(X, y)


# # # Predictions
# # predictions = model.predict(X)


# # # Model information

# # st.subheader("Model")

# # st.write("Coefficient:", model.coef_[0])
# # st.write("Intercept:", model.intercept_)

# # r2 = r2_score(y, predictions)
# # mae = mean_absolute_error(y, predictions)

# # st.write("R² Score:", r2)
# # st.write("MAE:", mae)


# # # Regression graph

# # st.subheader("Regression Line")

# # fig, ax = plt.subplots()

# # ax.scatter(X, y, label="Actual Data")
# # ax.plot(X, predictions, label="Regression Line")

# # ax.set_xlabel("Hours Studied")
# # ax.set_ylabel("Marks")
# # ax.legend()

# # st.pyplot(fig)



# # st.subheader("Predict Marks")

# # hours = st.number_input(
# #     "Enter hours studied",
# #     min_value=0.0,
# #     max_value=20.0,
# #     value=5.0
# # )
# # if st.button("Predict"):

# #     if hours >= 20:
# #         st.error("Hours must be less than 20")
# #     else:
# #         prediction = model.predict([[hours]])

# #         st.success(
# #             f"Predicted Marks: {prediction[0]:.2f}"
# #         )
