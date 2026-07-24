#classes

class MyClass:
    x=5


#Object creation for MyClass
p1 = MyClass()
#printing
print(p1.x)
#deleted object using del 
del p1
# print(p1.x)
# #error Traceback (most recent call last):
#   File "/Users/shaik/Desktop/MLLEARNINGS/OOPs.py", line 14, in <module>
#     print(p1.x)
#           ^^
# NameError: name 'p1' is not defined


#multiple Objects of P1
p1 = MyClass()
p2 = MyClass()
p3 = MyClass()

print(p1.x)
print(p2.x)
print(p3.x)


# class Person:
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age

#     def greet(self,name,age):
#         return f"Hello ,my name is {self.name}"




# p1 = Person("shaik",24)
# print(p1.greet())



class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("Emil", 36)

print(p1.name)
print(p1.age)



#without init we needed manually 
# Create a class without __init__():

class Person:
  pass

p1 = Person()
p1.name = "Tobias"
p1.age = 25

print(p1.name)
print(p1.age)


#with init 
class Person:
  def __init__(self, name, age, city, country):
    self.name = name
    self.age = age
    self.city = city
    self.country = country

p1 = Person("Linus", 30, "Oslo", "Norway")

print(p1.name)
print(p1.age)
print(p1.city)
print(p1.country)



class Dog:
  def __init__(self,name,age):
    self.name=name
    self.age=age
  def bark(self):
    print(f"says Woof!!!!{self.name}")



d1 = Dog("buddy",3)
d1.bark()



class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def greet(self):
    print("Hello, my name is " + self.name)

p1 = Person("Emil", 25)
p1.greet()


class Person:
  def __init__(self, name):
    self.name = name

  def printname(self):
    print(self.name)

p1 = Person("Tobias")
p2 = Person("Linus")

p1.printname()
p2.printname()



#self model of cars
class Car:
  def __init__(self,brand,model):
    self.brand=brand
    self.model=model

  def show(self):
    print(f"Your brand is {self.brand}")

  def ModelOfCar(self):
    print(f"Shown brand of car is {self.model}")



c1 = Car("Audi","124AE")
c2 = Car("TOYATA","HUHAJ82")
c2.show()
c1.show()
c1.ModelOfCar()



class Rectangle:
  def __init__(self,width,height):
    self.width = width
    self.height = height
 
  def area(self):
    return self.width*self.height


r1 = Rectangle(5,3)

print(r1.area())
