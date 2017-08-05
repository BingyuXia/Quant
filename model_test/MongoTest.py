import pymongo as mg
import pandas as pd
from WindPy import *

client = mg.MongoClient(host = 'localhost',port =  27017)
client.drop_database("test")
db = client.get_database(name ='paper')
col_01 = db.get_collection("col1")
data_01.insert({"a":2,"b":3})
data_01.insert({"b":6,"a":5})
data_01.remove({"a":2})
dt = col_01.find({"age":{"$lt": 20}},{"_id":0})
dt = col_01.find({"age":{"$in":[20]}})
dt = col_01.find({"age":{"$type":2}})
#collection01.insert({"a":"2"})
#for i in dt:
#    print(i)
dt2 = pd.DataFrame(list(dt)).set_index("age")
#dt2 = dt2.set_index("age")
print(dt2)





