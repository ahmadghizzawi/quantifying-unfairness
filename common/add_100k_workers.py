from dataset_100k import allwrkrs
from pymongo import MongoClient

client = MongoClient()
db = client.WorkerSet100K
collection = db.workers
db.collection_names(include_system_collections=False)
n = len(allwrkrs)

def InsertWorkers(n):
    for i in range(0, n):
        db.workers.insert_one(
            {
                "id": i,
                "Gender": allwrkrs[i][0],
                "Country": allwrkrs[i][1],
                "YearOfBirth": allwrkrs[i][2],
                "Language": allwrkrs[i][3],
                "Ethnicity": allwrkrs[i][4],
                "YearsOfExperience": allwrkrs[i][5],
                "LanguageTest": allwrkrs[i][6],
                "ApprovalRate": allwrkrs[i][7]
            })

InsertWorkers(int(n))

def read():
    empCol = db.workers.find()
    print('\n All data from EmployeeData Database \n')
    for emp in empCol:
        print(emp)
read()