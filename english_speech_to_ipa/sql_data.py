import os
import numpy as np
import pandas as pd
import sqlite3

from Errors import Error, DatabaseLimitError

class Connect_db:
    def __init__(self,database,tablename_ipa,tablename_mfcc,tablename_final):
        self.database = database
        self.table_ipa = tablename_ipa
        self.table_mfcc = tablename_mfcc
        self.table_final = tablename_final
        self.conn = sqlite3.connect(database)
        self.c = self.conn.cursor()

    def sqldata2df(self,tablename,column_value_list=None,limit=None,row_start=None):
        col_val = []
        if column_value_list:
            extra = " WHERE"
            for item in column_value_list:
                value = "'{}'".format(item[1])
                col_val.append('='.join(map(str, [item[0],value])))
        else:
            extra = ""
        if limit and row_start:
            if len(col_val) > 0:
                col_val[-1]+=(" LIMIT %s OFFSET %s" % (limit,row_start))
            else:
                col_val = [" LIMIT %s OFFSET %s" % (limit,row_start)]
        elif limit:
            if len(col_val) > 0:
                col_val[-1]+=(" LIMIT %s" % (limit))
            else:
                col_val.append(" LIMIT %s" % (limit))
        elif row_start:
            raise DatabaseLimitError("\nLimitMissingError: Need a LIMIT value in order to specify a ROWSTART value.\n")
            
        msg = ''' SELECT * FROM {}{} %s'''.format(tablename,extra) % (" AND ".join(col_val))
        self.c.execute(msg)
        data = self.c.fetchall()
        df = pd.DataFrame(data)
        return df
    
    def createsqltable(self,num_cols):
        #-1 because one column is already accounted for in "dataset" int
        columns = list((range(0,num_cols-1)))
        column_type = []
        for i in columns:
            column_type.append('"'+str(i)+'" real')
        msg = '''CREATE TABLE IF NOT EXISTS {}("dataset" int, %s)'''.format(self.table_final) % ", ".join(column_type)
        self.c.execute(msg)
        self.conn.commit()
        return None
    
    def databatch2sql(self,matrix):
        num_cols = matrix[0].shape[1] 
        self.createsqltable(num_cols)
        for i in range(len(matrix)):
            x = matrix[i]
            col_var = ""
            for j in range(num_cols):
                if j < num_cols-1:
                    col_var+=' ?,'
                else:
                    col_var+=' ?'
            msg = '''INSERT INTO {} VALUES (%s) '''.format(self.table_final) % col_var
            self.c.executemany(msg,x)
            self.conn.commit()
        return None
        
    
    def close_conn(self):
        if self.conn:
            self.conn.close()
        print("{} closed.".format(self.database))
        return None
