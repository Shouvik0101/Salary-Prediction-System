import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

def main():
    welcome()

def welcome ():
    print("******************************************************************************")
    print("**************************** Program Started *********************************")
    print("******************************************************************************")
    print("***** WELCOME TO SALARY PREDECTION SYSTEM BASED OF EMPLOYEES EXPERIENCES *****")
    print("******************************************************************************")
    print("Please press ENTER key to continue")
    input()


def checkcsv():
    get_dir=os.getcwd()
    get_files=os.listdir()
    store_csv=[]
    for X in get_files:
        if X.split(".")[-1]=="csv": # eg-> X contains "salary.csv"
            store_csv.append(X)     # break -> ["salary","csv"]
                                    #-1 is csv
    if len(store_csv)==0:
        return "no CSV file found"
    else:
        return store_csv

def display_and_print(csv_file):
    i=0
    for e in csv_file:
        print("File number: ",i," and file name: ",e)
        i=i+1
    return csv_file[int(input("Enter a file number that you want to choose: "))]
def graph(X_TRAIN,Y_TRAIN,Linear_Object,X_TEST,Y_PREDICT):
    plt.scatter(X_TRAIN,Y_TRAIN,label="training data",color="blue")
    plt.plot(X_TRAIN,Linear_Object.predict(X_TRAIN),label="best fit line",color="green")
    plt.scatter(X_TEST,Y_TEST,label="test data",color="yellow")
    plt.scatter(X_TEST,Y_PREDICT,label="predict data",color="blue")
    plt.xlabel("Year of Experiences")
    plt.ylabel("Salaries")
    plt.title("Exployee experiences vs Salary")
    plt.legend()
    plt.show()
if __name__=='__main__':
    main()

try:
    csv_files = checkcsv()
    if csv_files =="No CSV file found":
        raise FileNotFoundError
    select_csv = display_and_print(csv_files)
    print("Selected csv is: ", select_csv)
    print("Reading the csv file")
    load_csv=pd.read_csv(select_csv)
    print("CSV file is read and loa successfully")
    #print("some data of csv are")
    X=load_csv.iloc[:,:-1]
    Y=load_csv.iloc[:,-1]
    percent_train=float(input("How much percent data you want to keep for train(between 0 to 10): "))
    print("our dataset started with learning and training")
    X_TRAIN,X_TEST,Y_TRAIN,Y_TEST=train_test_split(X,Y,test_size=percent_train)
    print("Our data has been trined")
    print("Creating Linear Regression model")
    Linear_Object=LinearRegression()
    print("Our Linear Regression model has been created")
    print("Try to craete the best fit line with minimul errror")
    best_fit=Linear_Object.fit(X_TRAIN,Y_TRAIN)
    print("Our model is ready to test or predict unknow value based on X test data")
    print("Predicting the new target value based on Test Data")
    #print(X_TEST)
    Y_PREDICT=Linear_Object.predict(X_TEST)
    #print("following bwllow print X_TEST....Y_TEST....Y_PREDICT")
    graph(X_TRAIN,Y_TRAIN,Linear_Object,X_TEST,Y_PREDICT)
    accuresy=r2_score(Y_TEST,Y_PREDICT)
    print(f"Our model is accurate: {accuresy:2.2%}")
    print("Now our module is ready to predit salary based on new experinces")
    print("Please press ENTER to predict for new data")
    input()
    new_exp_list=[]
    i=1
    while True:
        print("Please enter a new expeinces for employee:",i)
        new_exp=float(input("enter here -> "))
        new_exp_list.append(new_exp)
        i=i+1
        ans=input("Do you want to add more eployee experinecs(y/n)")
        if ans!="y":
            break
    input_target=pd.DataFrame({
        "  Year of Experiences(in year)":new_exp_list
    })
    print("New list of employees experiences ", input_target)
    Y_target_predict=Linear_Object.predict(input_target)  
    print("Our model has predicted the target data for new experinces")
    print("Plotting graphical representation for new target data")
    plt.scatter(input_target,Y_target_predict,label="new target data",color="green")
    plt.title("New expeiences VS New salary")
    plt.xlabel("Year of Experinecs")
    plt.ylabel("Salaries")
    plt.legend()
    plt.show()
    new_target=pd.DataFrame({
        "Experinecs":new_exp_list,
        "Salary":Y_target_predict
    })
    print(new_target)
except FileNotFoundError:
    print("No CSV file found")
    print("Our model has does not learned, trained and predicting new target data successfully")
except Exception as exp:
    print("Other issue ",exp)
    print("Our model has does not learned, trained and predicting new target data successfully")
else:
    print("******************************************************************************")
    print("* Our model has learned, trained and predicting new target data successfully *")
    print("******************************************************************************")
finally:
    print("************************ Program Ended ***************************************")
