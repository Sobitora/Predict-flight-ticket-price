import xlsxwriter
workbook = xlsxwriter.Workbook('input.xlsx')
worksheet = workbook.add_worksheet()

bold = workbook.add_format({'bold': True})

info  = []
info.append(input("Airline : "))
info.append(input("Date (6/06/2019) : "))
info.append(input("Source : "))
info.append(input("Destination : "))
info.append(input("Time (17:30) : "))
info.append(input("Duration time (ex. 10h 55m) : "))
info.append(input("Total_Stops : "))
info.append(input("Additional_Info : "))

section = ['Airline','Date_of_Journey','Source','Destination','Dep_Time','Duration','Total_Stops','Additional_Info']
row = 0
column = 0
for i in section:
    worksheet.write(row,column,i,bold)
    column += 1


row = 1
column = 0


for i in info:
    worksheet.write(row,column,i)
    column += 1
workbook.close()