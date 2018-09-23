import os
import numpy as np
import pandas as pd
import sqlite3
from sqlite3 import Error

class Error(Exception):
    """Base class for other exceptions"""
    pass

class LimitMissingError(Error):
   """If rowstart is specified but limit is not"""
   pass

class Connect_db:
    def __init__(self,database,tablename):
        self.database = database
        self.table = tablename
        self.conn = sqlite3.connect(database)
        self.c = self.conn.cursor()

    def sqldata2df(self,column_value_list=None,limit=None,row_start=None):
        col_val = []
        if column_value_list:
            extra = " WHERE"
            for item in column_value_list:
                item[1] = "'{}'".format(item[1])
                col_val.append('='.join(map(str, item)))
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
            raise LimitMissingError("\nLimitMissingError: Need a LIMIT value in order to specify a ROWSTART value.\n")
            
        msg = ''' SELECT * FROM {}{} %s'''.format(self.table,extra) % (" AND ".join(col_val))
        print(msg)
        self.c.execute(msg)
        data = self.c.fetchall()
        df = pd.DataFrame(data)
        return df
