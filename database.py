# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:37:23 2019

@author: shivansh
"""

import sqlite3
conn = sqlite3.connect('model_result.db')
print ("Opened database successfully")

#conn.execute("ALTER TABLE RESULTS RENAME TO old");
conn.execute('''CREATE TABLE RESULTS (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                COMMENT BLOB NOT NULL,
                RESULT REAL NOT NULL)''');
 

conn.commit()
print ('Record created successfully')
conn.close()