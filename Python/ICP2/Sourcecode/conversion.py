N= input('Enter a list of Numbers seperated by space:')
a=0
N1=[]
userList=N.split()
print(userList)

N1=[float(i)*0.4535 for i in userList]
print('Below are  10 the list of numbers converted from lbs to kgs')
print(N1)