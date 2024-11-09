import csv
from datetime import datetime

################################################################
# GROUP MEMBERS: 
#   - Aldrin Amistoso
#   - Marc Angeles
#   - Marvin Estrada
#   - Jesse Garcia
# ASGT: Project
# ORGN: CSUB - CMPS 3500 
# FILE: ClassProjectGroup3.py
# DATE: 11/31/2024
################################################################

def loadData(filename):
    # Record start time to measure load time
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Script")
    
    # Initialize counters for rows and columns
    row_count = 0
    col_count = 0
    
    # Load data
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading training data set")
    try:
        # Open csv file for reading
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)

            # read header row to get column count
            headers = next(csv_reader)  
            col_count = len(headers)

            # iterate over each row to count total rows
            for row in csv_reader:
                row_count += 1

        # print total columns and rows read
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total Columns Read: {col_count}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total Rows Read: {row_count}")
    
    except FileNotFoundError:
        print("Error: File not found. Please ensure the file exists in the specified path.")
        return
    
    # load time calculation
    load_time = datetime.now() - start_time
    load_time_seconds = load_time.total_seconds()
    print(f"\nTime to load is: {load_time_seconds:.2f} seconds\n")

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
            print("\nLoading and Cleaning Input data set:")
            print("************************************")
            loadData("credit_score_data.csv")
        
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
