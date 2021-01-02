import sqlite3

p = sqlite3.connect("file::memory:?cache=shared", uri=True)
p.execute('CREATE TABLE tbTest (fld1, fld2)')
p.execute("INSERT INTO tbTest VALUES ('fld1', 'fld2')")
p.commit()
t = list(p.execute('select * from tbTest '))
print(t)
