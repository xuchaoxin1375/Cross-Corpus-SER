# # a=1
# # b=a
# # print(f"{a=} {b=}; {id(a)=} {id(b)=}")
# # a=100
# # print(f"{a=} {b=}; {id(a)=} {id(b)=}")
# # test=100
# # print(f"{test=} {id(test)=}")

# ##
# print(f"{id(10)=}")
# a=10
# print(f"{id(a)=}")

# def f(x):
#     print(f"{id(x)=}")
#     x=20
#     print(f"{id(20)=}")
#     print(f"{id(x)=}")

# f(a)
# a
# print('a: ', a)#10
a=[1,2,3]
def f(x):
	x=100
f(a)
print(f"{a=}")#[1,2,3]

b=[1,2,3]
def g(x):
	x[0]=100
g(b)
print(f"{b=}")

