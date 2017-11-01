col1=158
col2=11
col_zer = 10
plt.figure()


plt.scatter(data[col1][data_y[0] == 1],
            data[col2][data_y[0] == 1],
            alpha=0.75,
            color='red',
            label='1')

##plt.scatter(data_norm[col1][data_y[0] == 2],
##            data_norm[col2][data_y[0] == 2],
##            alpha=0.75,
##            color='blue',
##            label='2')
##plt.scatter(data_norm[col1][data_y[0] == 3],
##            data_norm[col2][data_y[0] == 3],
##            alpha=0.75,
##            color='yellow',
##            label='3')
##plt.scatter(data_norm[col1][data_y[0] == 4],
##            data_norm[col2][data_y[0] == 4],
##            alpha=0.75,
##            color='green',
##            label='4')
##plt.scatter(data_norm[col1][data_y[0] == 5],
##            data_norm[col2][data_y[0] == 5],
##            alpha=0.75,
##            color='black',
##            label='5')
plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(loc='best')
plt.show()
