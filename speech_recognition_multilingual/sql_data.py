import os
import numpy as np
import pandas as pd

class Error(Exception):
    pass

class LimitMissingError(Error):
   """Base class for other exceptions"""
   pass


def sqldata2df(c,tablename,column_value_list,limit=None,row_start=None):
    try:
        col_val = []
        for item in column_value_list:
            item[1] = "'{}'".format(item[1])
            col_val.append('='.join(map(str, item)))
        if limit and row_start:
            col_val[-1]+=(" LIMIT %s OFFSET %s" % (limit,row_start))
        elif limit:
            col_val[-1]+=(" LIMIT %s" % (limit))
        elif row_start:
            raise LimitMissingError
            
        msg = ''' SELECT * FROM {} WHERE %s'''.format(tablename) % (" AND ".join(col_val))
        print(msg)
        c.execute(msg)
        data = c.fetchall()
        df = pd.DataFrame(data)
        return df
    except LimitMissingError:
        print("\nLimitMissingError: Need a LIMIT value in order to specify a ROWSTART value.\n")




