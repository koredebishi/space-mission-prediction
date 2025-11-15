"""
#Binary seach takes a sorted list and return a 
# the position of the list in the list. 
#[1 , 2, 3, 4, 5, 6] 
def binary_search(lst, item):
    low = 0      # [1]
    high = len(list) - 1  # [6]
    
    while low <= high:

        mid = (high - low)  #[3]

        guess = lst[mid]    # [3] ,, I think the item is @ 3 , item = 6

        if guess == item:
            return mid
        if guess < item:
            low = mid + 1
        else:  high = mid - 1
    return None    
        

list = [1 , 2, 3, 4, 6, 7]
item = 10

result = binary_search(list, item)

#print("found item @ ", result)

"""

"""
Given a list of length N, return true or false if the item is in the list. 


"""
import time

# def search (list, item):

#     #return item in list # rund O(2n)
#     # if item in list:      
#     #     return list.index(item) == True
#     # else:
#     #     return False 
    
#     for i , val in enumerate(list):   # runs O(n)
        
#         if val == item:
#             return i
#     return False

# #Given an array list, return if a number appears more than 2x in the list.
# def search (list, item):
#     count = 0
#     for val in list:     # O(n)
#         if val == item:  # comparism 
#             count +=1    # operation
#             if count > 1:
#                 return True
#     return False        
              
# list = [1, 1, 2,3 ,5, 6, 1]
# item = 1


#Given an array of number in the range [0, n]. return a missing number in the range
#Using set



list = [0, 1, 3, 4]
item = 2
result = search(list, item)

print("item is : ", result)
