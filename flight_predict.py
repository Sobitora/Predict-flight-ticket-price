# UI
import tkinter as tk
from tkinter import  Frame, Spinbox, ttk
from tkcalendar import DateEntry
root = tk.Tk()
root.title("Predict")
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=2)
root.columnconfigure(2, weight=1)
root.columnconfigure(3, weight=2)
root.columnconfigure(4, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)
root.rowconfigure(6, weight=1)
root.rowconfigure(7, weight=1)

inp = []
label = tk.Label(text='Airline',font=26).grid(row=1,column=0,sticky=tk.EW,padx=10)
inpAir = tk.StringVar()
Airlinechoosen = ttk.Combobox( width = 27, textvariable = inpAir,state='readonly')
Airlinechoosen['values'] = ('IndiGo', 
                        'Air India',
                        'Jet Airways',
                        'SpiceJet',
                        'Multiple carriers',
                        'GoAir', 
                        'Vistara', )

Airlinechoosen.grid(column = 1, row = 1)



label = tk.Label(text='Dep_Time',font=26).grid(row=1,column=2,sticky=tk.EW,padx=10)
timeframe = Frame(root)
inphour = tk.StringVar()
inpmin = tk.StringVar()
hr_inp = Spinbox(timeframe,from_=0,to=23,textvariable=inphour).grid(row=0,column=0)
min_inp = Spinbox(timeframe,from_=0,to=59,textvariable=inpmin).grid(row=0,column=1)
inptime = inphour.get()+":"+inpmin.get()
timeframe.grid(row=1,column=3)

label = tk.Label(text='Date_of_Journey',font=26).grid(row=2,column=0,sticky=tk.EW,padx=10)
inpDate = tk.StringVar()
cal = DateEntry(root, width=12, year=2019, month=6, day=22,background='black', foreground='white', borderwidth=2,state='readonly',date_pattern="d/mm/y",textvarible=inpDate)
cal.grid(row=2,column=1,sticky=tk.EW)



label = tk.Label(text='Duration',font=26).grid(row=2,column=2,sticky=tk.EW,padx=10)
DurationFrame = Frame(root)
inp_h = tk.StringVar()
inp_m = tk.StringVar()
h_inp = tk.Entry(DurationFrame,textvariable=inp_h).grid(row=0,column=0)
h_text = tk.Label(DurationFrame,text="h",font=26).grid(row=0,column=1)
m_inp = tk.Entry(DurationFrame,textvariable=inp_m).grid(row=0,column=2)
m_text = tk.Label(DurationFrame,text="m",font=26).grid(row=0,column=3)
DurationFrame.grid(row=2,column=3)
inpDura = inp_h.get()+"h"+inp_m.get()+"m"

label = tk.Label(text='Source',font=26).grid(row=3,column=0,sticky=tk.EW,padx=10)
inpSource = tk.StringVar()
Sourcechoosen = ttk.Combobox( width = 27, textvariable = inpSource,state='readonly')
Sourcechoosen['values'] = ('Banglore', 
                        'Kolkata',
                        'Delhi',
                        'Chennai',
                        'Mumbai',)

Sourcechoosen.grid(row=3,column=1,sticky=tk.EW)


label = tk.Label(text='Total_Stops',font=26).grid(row=3,column=2,sticky=tk.EW,padx=10)
inpStop = tk.StringVar()
Stopchoosen = ttk.Combobox( width = 27, textvariable = inpStop,state='readonly')
Stopchoosen['values'] = ('non-stop', 
                        '1 stop',
                        '2 stops',
                        '3 stops',
                        '4 stops',)

Stopchoosen.grid(row=3,column=3,sticky=tk.EW)

label = tk.Label(text='Destination',font=26).grid(row=4,column=0,sticky=tk.EW,padx=10)
inpDes = tk.StringVar()
Deschoosen = ttk.Combobox( width = 27, textvariable = inpDes,state='readonly')
Deschoosen['values'] = ('New Delhi', 
                        'Delhi',
                        'Banglore',
                        'Cochin',
                        'Kolkata',
                        'Hyderabad')
Deschoosen.grid(row=4,column=1,sticky=tk.EW)


label = tk.Label(text='Additional_Info',font=26).grid(row=4,column=2,sticky=tk.EW,padx=10)
inpInfo = tk.StringVar()
Infochoosen = ttk.Combobox( width = 27, textvariable = inpInfo,state='readonly')
Infochoosen['values'] = ('No info', 
                        'In-flight meal not included',
                        'No check-in baggage included',
                        'Change airports',
                        'Business class',
                        '2 Long layover',
                        '1 Long layover')
Infochoosen.grid(row=4,column=3,sticky=tk.EW)

def saveinput():
    
    inp.append(inpAir.get())
    inp.append(cal.get_date().strftime("%m/%d/%Y"))
    inp.append(inpSource.get())
    inp.append(inpDes.get())
    inp.append(inphour.get()+":"+inpmin.get())
    inp.append(inp_h.get()+"h"+inp_m.get()+"m")
    inp.append(inpStop.get())
    inp.append(inpInfo.get())
    import pandas as pd
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    sns.set_style("whitegrid")

    flights=pd.read_excel('./Data_Train.xlsx')

    flights.dropna(inplace=True)

    flights['Date_of_Journey']=pd.to_datetime(flights['Date_of_Journey'],dayfirst=True)
    flights['Dep_Time']=pd.to_datetime(flights['Dep_Time'],format='%H:%M').dt.time
    flights['Additional_Info']=flights['Additional_Info'].str.replace('No info','No Info')
    flights['Duration']=flights['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
    flights['Duration']=pd.to_numeric(flights['Duration'])
    flights['weekday']=flights[['Date_of_Journey']].apply(lambda x:x.dt.day_name()) 
    flights["month"] = flights['Date_of_Journey'].map(lambda x: x.month_name()) 
    flights['Dep_Time']=flights['Dep_Time'].apply(lambda x:x.hour)
    flights['Dep_Time']=pd.to_numeric(flights['Dep_Time'])


    flights.drop(['Route','Arrival_Time','Date_of_Journey'],axis=1,inplace=True)

    var_mod = ['Airline','Source','Destination','Additional_Info','Total_Stops','weekday','month','Dep_Time']
    le = LabelEncoder()
    for i in var_mod:
        flights[i] = le.fit_transform(flights[i])

    # outlier
    def outlier(df):
        for i in df.describe().columns:
            Q1=df.describe().at['25%',i]
            Q3=df.describe().at['75%',i]
            IQR= Q3-Q1
            LE=Q1-1.5*IQR
            UE=Q3+1.5*IQR
            df[i]=df[i].mask(df[i]<LE,LE)
            df[i]=df[i].mask(df[i]>UE,UE)
        return df
    flights=outlier(flights)
    x=flights.drop('Price',axis=1)
    y=flights['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

    rfr=RandomForestRegressor(n_estimators=500) #random data 500 item
    rfr.fit(x_train,y_train)

    import xlsxwriter
    workbook = xlsxwriter.Workbook('input.xlsx')
    worksheet = workbook.add_worksheet()

    bold = workbook.add_format({'bold': True})

    section = ['Airline','Date_of_Journey','Source','Destination','Dep_Time','Duration','Total_Stops','Additional_Info']
    row = 0
    column = 0
    for i in section:
        worksheet.write(row,column,i,bold)
        column += 1


    row = 1
    column = 0
    for i in inp:
        worksheet.write(row,column,i)
        column += 1
    workbook.close()

    user_input=pd.read_excel('input.xlsx')
    user_input['Date_of_Journey']=pd.to_datetime(user_input['Date_of_Journey'],dayfirst=True)
    user_input['Dep_Time']=pd.to_datetime(user_input['Dep_Time'],format='%H:%M').dt.time
    user_input['Duration']=user_input['Duration'].str.replace('h','*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
    user_input['Duration']=pd.to_numeric(user_input['Duration'])
    user_input['Dep_Time']=user_input['Dep_Time'].apply(lambda x:x.hour)
    user_input['Dep_Time']=pd.to_numeric(user_input['Dep_Time'])
    user_input["month"] = user_input['Date_of_Journey'].map(lambda x: x.month_name())
    user_input['weekday']=user_input[['Date_of_Journey']].apply(lambda x:x.dt.day_name())
    user_input['Additional_Info']=user_input['Additional_Info'].str.replace('No info','No Info')
    user_input.drop(['Date_of_Journey'],axis=1,inplace=True)
    for i in var_mod:
        user_input[i]=le.fit_transform(user_input[i])
    test_price_predictions=rfr.predict(user_input)
    price = Frame(root)
    label = tk.Label(price,text="Price : ",font=26).grid(row=0,column=0)
    label = tk.Label(price,text=test_price_predictions[0],font=26).grid(row=0,column=1)
    price.grid(row=5,column=0,columnspan=5)
    inp.clear()

price = Frame(root)
label = tk.Label(price,text="Price : ",font=26).grid(row=0,column=0)
label = tk.Label(price,text="",font=26).grid(row=0,column=1)
price.grid(row=5,column=0,columnspan=5)
btn = tk.Button(text="Save",width=10,command=saveinput).grid(row=6,column=0,columnspan=5)




root.mainloop()