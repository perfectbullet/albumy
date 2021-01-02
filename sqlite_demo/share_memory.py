import sqlite3

q = sqlite3.connect("file:memDB1?mode=memory&cache=shared")
# q.execute("INSERT INTO tbTest VALUES ('fld3', 'fld4')")
# q.commit()

t = list(q.execute('select * from tbTest '))
print(t)
