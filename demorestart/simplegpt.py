import numpy as np

def softmax(x):
    x = x-np.max(x)
    return np.exp(x)/sum(np.exp(x))

res = np.array([-3.2,0.6,-0.9,1.1,256])

print(softmax(res))



#practicing classes



def check(a,b):
    return a+b;

result = check(10,20)
print(result)


class Student:
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"my name is {self.name} and my age is {self.age}.")

    print("="*50)
    def checkeven(self,n):
        if(n%2==0):
            print(f"this is even number {self.name}")
        else:
            print(f"this is Odd number {self.name}")



result = Student("shaik",23)

result.introduce()

result.checkeven(11)



class Car:
    def __init__(self,model,brand):
        self.model = model
        self.brand = brand

    def display(self):
        print(f"Car brand is {self.brand} and model {self.model}")



carmodel = Car("S-series","BMW")

carmodel.display()


#lists
# List
students = ["Ravi", "Priya", "Fazil"]
students.append("Arun")

print(students)


# Dictionary
student = {
    "name": "Fazil",
    "age": 23,
    "marks": 90
}

print(student["name"])
print(student["marks"])


# File handling
with open("student.txt", "w") as file:
    file.write("Fazil scored 90 marks.")


with open("student.txt",'+a') as file:
    file.writelines("Hey this side fazil i had scored a good marks")

    
with open("student.txt", "r") as file:
    content = file.read()

print(content)

print("*"*50)
print("Numpy array")
print("*"*50)

#numpy arrays

import numpy as np

numbers = np.array([10,20,30,40])

print("Original array:", numbers)
print("Add 5:", numbers + 5)
print("Multiply by 2:", numbers * 2)
print("Average:", np.mean(numbers))

numpyarray = np.array([2,4,6,8,10])

print(numpyarray*3)


print("mean and max of array ")
mean_array = [15, 25, 35, 45, 55]

print(np.mean(mean_array))
print(np.max(mean_array))


#processing text documents

from collections import Counter

with open('reviews.txt','r') as file:
    content = file.read()


text = content.lower()

words = content.split()

word_count = Counter(words)

print(word_count)