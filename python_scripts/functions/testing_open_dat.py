# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:13:00 2019

@author: matheus.ladvig
"""

import numpy as np
from pathlib import Path
from scipy import stats
from scipy import interpolate

file = (Path(__file__).parents[3]).joinpath('data_PIV').joinpath('wake_Re300_export_190709_0100').joinpath('B0001.dat')


data = open(str(file))



datContent = [i.strip().split() for i in data.readlines()]

data = datContent[4:]

nb_lines = len(data)
nb_collums = len(data[0])

matrix = np.zeros(shape=(nb_lines,nb_collums))

for i,line in enumerate(data):
    for j,number in enumerate(line):
        matrix[i,j] = number

##################################################### speed in m/s
        
        
#        |x(mm)|y(mm)|vx(m/s)|vy(m/s)|isValid|
        
'''

- Normalise velocity per infinity velocity
- Normalise grid per cilinder diameter

'''

u_inf_measured = 0.388 # m/s
cil_diameter = 12 # 12mm


grid = matrix[:,0:2] 
#data = matrix[:,2:4] 
valid = matrix[:,4] # Get the effective values
matrix_valid_grid = matrix[np.where((valid==1))[0],:] # Select only the points that have a value
matrix_valid_grid[:,2:4] = matrix_valid_grid[:,2:4]/u_inf_measured
############### PIV centering and unity of measure to 1/D  #########################

'''
This PIV data coordiantes are ((xo ; yo) = (-75.60 ; 0.75) )
'''


matrix_valid_grid_new_coord_x = (matrix_valid_grid[:,0] + 75.60)/cil_diameter
matrix_valid_grid_new_coord_y = (matrix_valid_grid[:,1] - 0.75)/cil_diameter
matrix_valid_grid[:,0] = (matrix_valid_grid[:,0] + 75.60)/cil_diameter
matrix_valid_grid[:,1] = (matrix_valid_grid[:,1] - 0.75)/cil_diameter
'''
- Matrix_valid_grid_new_coord_x is the x distance vector from the points to the center of the cylinder
- Matrix_valid_grid_new_coord_y is the y distance vector from the points to the center of the cylinder


'''

##################################### DNS centering ####################################

'''
- The center of the cilynder is the position 60 in the vector x and 49 in vector y and we must centralise the coordinates beggining in these points 

'''


grid_dns = np.load('grid.npy')
topos_dns = np.load('topos.npy')

topos_dns = np.reshape(topos_dns,newshape =(422,97,76,7,3),order='F')

grid_dns_x = grid_dns[0]
grid_dns_y = grid_dns[1]


grid_dns_x_centralised = grid_dns_x[:,0] - grid_dns_x[60,0]
grid_dns_y_centralised = grid_dns_y[:,0] - grid_dns_y[49,0]


################################--Cutting less accurate pixels of PIV--#######################################################
#spatial_period_x = matrix_valid_grid_new_coord_x[1] - matrix_valid_grid_new_coord_x[0]
#spatial_period_y = matrix_valid_grid_new_coord_y[0] - matrix_valid_grid_new_coord_y[34]

'''
The grid is variable because the algorithm of PIV dont give us a square grid. Then We must find the point to cut and transform it in a 
squared grid to posterior 2d interpolations 

'''

end_points = []
start_points = []


start_points.append([0,matrix_valid_grid_new_coord_x[0]])
for i,value in enumerate(matrix_valid_grid_new_coord_x[:-1]):
    if (matrix_valid_grid_new_coord_x[i+1]-value<0):
        end_points.append([i,value])
        start_points.append([i+1,matrix_valid_grid_new_coord_x[i+1]])

end_points.append([i+1,matrix_valid_grid_new_coord_x[i+1]])


'''
- The first line of x and te last one seems to be always corrupted, so i'll cut off the first and the last x grid line

- The vector matrix_valid_grid_new_coord_x will begin in 34(The first 34 are samples are incomplete)
- The vector will end in 28171(The last line of x is incomplete).
'''

matrix_valid_grid = matrix_valid_grid[start_points[1][0]:end_points[-2][0]+1]

matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[start_points[1][0]:end_points[-2][0]+1]
matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[start_points[1][0]:end_points[-2][0]+1]
start_points = start_points[1:-1]
end_points = end_points[1:-1]

'''
Now it's necessary to find the squared grid 

'''
start_points = np.array(start_points)
end_points = np.array(end_points)


start_x = np.max(start_points[:,1])
end_x = np.min(end_points[:,1])


indexes_of_grid = np.where((matrix_valid_grid_new_coord_x>=start_x)&(matrix_valid_grid_new_coord_x<=end_x))



'''
Selecting only the elements inside the grid with limits [start_x, end_x] --> the results is a squared grid 
'''
matrix_valid_grid = matrix_valid_grid[indexes_of_grid]

matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_of_grid]
matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_of_grid]


########################### 2nd part ########################
'''
Cut 'n'(3,4,5...????) pixels from the PIV squared grid ---> Estimation algorithm less accurate in the window boundary 

'''

n = 3

value_in_n_start = matrix_valid_grid_new_coord_x[n]
value_in_n_end = matrix_valid_grid_new_coord_x[-n-1]


indexes_of_grid_n = np.where((matrix_valid_grid_new_coord_x>=value_in_n_start)&(matrix_valid_grid_new_coord_x<=value_in_n_end))
matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_of_grid_n]
matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_of_grid_n]
matrix_valid_grid = matrix_valid_grid[indexes_of_grid_n]

######################### 3rd part  ####################
'''
The cilynder is not centralised in the window,(i.e the number of sampled lines in left is different from the right of the cilynder), we must centralise it, selecting 
the same number in both sides and a sample to be the centered(The nearest sample to 0). 
'''

sampled_points_in_y = np.unique(matrix_valid_grid_new_coord_y)
indexes_pos = np.where((sampled_points_in_y>0))[0]
indexes_neg = np.where((sampled_points_in_y<0))[0]
positive = len(indexes_pos)
negative = len(indexes_neg)

if positive>negative:
    indexes_pos = indexes_pos[:negative]
    sampled_points_in_y = sampled_points_in_y[np.concatenate((indexes_neg,indexes_pos))]
   
    if sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]<0:
        sampled_points_in_y = sampled_points_in_y[:-1]
    elif sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]>0:
        sampled_points_in_y = sampled_points_in_y[1:]
    
    indexes_after_find_equal_sides = np.where((matrix_valid_grid_new_coord_y<=sampled_points_in_y[-1]))
    matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_after_find_equal_sides]
    matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_after_find_equal_sides]
    matrix_valid_grid = matrix_valid_grid[indexes_after_find_equal_sides]
elif positive<negative:
    indexes_neg = indexes_neg[:positive]
    sampled_points_in_y = sampled_points_in_y[np.concatenate((indexes_neg,indexes_pos))]
    if sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]<0:
        sampled_points_in_y = sampled_points_in_y[:-1]
    elif sampled_points_in_y[np.argmin(np.abs(sampled_points_in_y))]>0:
        sampled_points_in_y = sampled_points_in_y[1:]
    
    
    indexes_after_find_equal_sides = np.where((matrix_valid_grid_new_coord_y>=sampled_points_in_y[0]))
    matrix_valid_grid_new_coord_x = matrix_valid_grid_new_coord_x[indexes_after_find_equal_sides]
    matrix_valid_grid_new_coord_y = matrix_valid_grid_new_coord_y[indexes_after_find_equal_sides]
    matrix_valid_grid = matrix_valid_grid[indexes_after_find_equal_sides]

'''
In this line, the vector 'sampled_points_in_y' represents the sampled points in y with the same quantity in both sides 

'''


##################### 4th part  ###################

'''
Here, we begin to process the DNS grid, first we need to centralise the cilynder in the middle of the window DNS grid as we did with PIV grid

''' 

indexes_pos = np.where((grid_dns_y_centralised>0))[0]
indexes_neg = np.where((grid_dns_y_centralised<0))[0]
positive = len(indexes_pos)
negative = len(indexes_neg)

if positive>negative:
    indexes_pos = indexes_pos[:negative]

elif positive<negative:
    indexes_neg = indexes_neg[int(negative-positive):]


grid_dns_y_centralised = grid_dns_y_centralised[np.concatenate((indexes_neg,np.array([indexes_neg[-1]+1]),indexes_pos))]
topos_dns = topos_dns[:,np.concatenate((indexes_neg,np.array([indexes_neg[-1]+1]),indexes_pos)),:,:,:]



################### 5 th part 
x_piv = np.unique(matrix_valid_grid_new_coord_x)
y_piv = np.unique(matrix_valid_grid_new_coord_y)


'''
The dimensions of PIV and DNS are not the same, therefore we must find the same number of sampled points for each dimension
PIV_range ---->  x = (0.74,10.44) y=(-2.84,2.83)
DNS_range ---->  x = (-2.5,15.04) y=(-1.95,1.95)

Therefore, the effective window(where we have information about model and obs) is the intersection between both ones.  

        
                
                ----------------------------       -   -    -    -
                |                          |                     |
    ------------------------------------------------    -
    |        ---                           |       |    |        |
    |       -----                          |       |   3.9      5.68
    |        ---                           |       |    |        |
    ------------------------------------------------    -           ---------------->DNS window
                |                          |                     |
                ----------------------------      -    -    -    -  ---------------->PIV window
    
                |<    -  -   11.18   -  - >|         
    
    
    |<     -       -         17.19      -     -   >|


'''


first_x_PIV = x_piv[0]
last_x_PIV = x_piv[-1]
index_min = np.where((grid_dns_x_centralised>first_x_PIV))[0][0]
index_max = np.where((grid_dns_x_centralised>last_x_PIV))[0][0]


grid_dns_x_centralised = grid_dns_x_centralised[index_min-1:index_max+1]
topos_dns = topos_dns[index_min-1:index_max+1,...]

##########

extreme_left_DNS = grid_dns_y_centralised[0]
extreme_right_DNS = grid_dns_y_centralised[-1]

index_min = np.where((y_piv<extreme_left_DNS))[0][-1]
index_max = np.where((y_piv>extreme_right_DNS))[0][0]


y_piv = y_piv[index_min+1:index_max]


'''
Now selecting the points in matrix valid grid that the collum of y is the same of y_piv
'''

indexes_matrix_valid_grid = np.where((matrix_valid_grid[:,1]>=y_piv[0])&(matrix_valid_grid[:,1]<=y_piv[-1]))[0]

matrix_valid_grid = matrix_valid_grid[indexes_matrix_valid_grid,:]


####################################################################

'''
Now it's necessary :
    1°)  Interpolate Topos inside the grid grid_dns_x_centralised and grid_dns_y_centralised to the grid of x_piv,y_piv
        
    Topos are already selected in the points after pre-processing(centralizing, cutting in x), but the points in space are not the same and are not in the same quantity.
    Because the spatial sampling frequency is not the same. 
'''


topos_dns = topos_dns[:,:,30,:,:2]
topos_new_coordinates = np.zeros(shape=(len(x_piv),len(y_piv),*topos_dns.shape[2:]))
x = grid_dns_x_centralised
y = grid_dns_y_centralised




for phi in range(topos_dns.shape[2]):
    for dim in range(topos_dns.shape[3]):
        z = topos_dns[:,:,phi,dim].T
        f = interpolate.interp2d(x, y, z, kind='cubic')
        znew = f(x_piv, y_piv)
        topos_new_coordinates[:,:,phi,dim] = znew.T






































