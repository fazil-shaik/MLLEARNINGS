import numpy as np
import pandas as pd

np.random.seed(42)

rows = 300

data = {
    "ExperienceYears": np.random.randint(0, 21, rows),
    "EducationLevel": np.random.randint(1, 6, rows),
    "SkillScore": np.random.randint(40, 101, rows),
    "Certifications": np.random.randint(0, 11, rows),
    "Age": np.random.randint(21, 56, rows),
    "CommuteDistance": np.random.randint(5, 71, rows),
}

df = pd.DataFrame(data)

# Salary formula (realistic correlation)
df["Salary"] = (
    20000
    + df["ExperienceYears"] * 3500
    + df["EducationLevel"] * 5000
    + df["SkillScore"] * 600
    + df["Certifications"] * 2500
    - df["CommuteDistance"] * 50
    + np.random.randint(-5000, 5000, rows)
).astype(int)

df.to_csv("salary_300_rows.csv", index=False)
print(df.head())
