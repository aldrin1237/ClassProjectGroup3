################################################################
# GROUP MEMBERS: 
#   - Aldrin Amistoso
#   -
#   - 
#   - 
# ASGT: Project
# ORGN: CSUB - CMPS 3500 
# FILE: ClassProjectGroup3.py
# DATE: 11/31/2024
################################################################

def mainMenu():
    while True:
        print("===============================================")
        print("\nMenu:")
        print("(1) Load Data")
        print("(2) Clean Data")
        print("(3) Train Model")
        print("(4) Test Model")
        print("(5) Exit")
        
        choice = input("Select Option: ")
        print("===============================================")
        if choice == '1':
            print("\nLoading and Cleaning Input data set")
            print("************************************")
            print(" Starting Script")
            print(" Loading training data set")
            print(" Total Columns Read:")
            print(" Total Rows Read:")
            print("\nTime to load is:  ")

        
        elif choice == '2':
            print("\nProcessing input data set:")
            print("**************************")
            print(" Performing Data Clean Up")
            print(" Total Rows after cleaning is: ")
            print("\nTime to process is: ")
    

        elif choice == '3':
            print("\nPrinting Model details:")
            print("***********************")
            print(" Model RMSE ")
            print(" Hyper-parameters used are:")
        
        elif choice == '4':
            print("\nTesting Model:")
            print("**************")
            print(" Generating prediction using selected Neural Network")
            print(" Size of training set ")
            print(" Size of testing set ")
            print(" Predictions generated ... ")
        
        elif choice == '5':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid option. Please select a valid option.")

if __name__ == "__main__":
    mainMenu()
