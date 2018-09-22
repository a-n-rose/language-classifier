import os
import numpy as np
import pandas as pd

class Error(Exception):
    """Base class for other exceptions"""
    pass

class LimitMissingError(Error):
   """If rowstart is specified but limit is not"""
   pass


def sqldata2df(c,tablename,column_value_list=None,limit=None,row_start=None):
    try:
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
            raise LimitMissingError
            
        msg = ''' SELECT * FROM {}{} %s'''.format(tablename,extra) % (" AND ".join(col_val))
        print(msg)
        c.execute(msg)
        data = c.fetchall()
        df = pd.DataFrame(data)
        return df
    except LimitMissingError:
        print("\nLimitMissingError: Need a LIMIT value in order to specify a ROWSTART value.\n")
