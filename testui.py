import tkinter as tk
from tkinter import  Frame, Spinbox, ttk
from tkcalendar import DateEntry
from tktimepicker import AnalogPicker, AnalogThemes
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
inp = []
# info.append(tk.Entry().grid(row=0,column=0))
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
Airlinechoosen.current(0) 



label = tk.Label(text='Dep_Time',font=26).grid(row=1,column=2,sticky=tk.EW,padx=10)
# inptime = tk.StringVar()
timeframe = Frame(root)
inphour = tk.StringVar()
inpmin = tk.StringVar()
hr_inp = Spinbox(timeframe,from_=0,to=23,textvariable=inphour).grid(row=0,column=0)
min_inp = Spinbox(timeframe,from_=0,to=59,textvariable=inpmin).grid(row=0,column=1)
inptime = inphour.get()+":"+inpmin.get()
timeframe.grid(row=1,column=3)
# text = tk.Entry().grid(row=0,column=3,sticky=tk.EW)

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
# text = tk.Entry().grid(row=2,column=3,sticky=tk.EW)

label = tk.Label(text='Source',font=26).grid(row=3,column=0,sticky=tk.EW,padx=10)
inpSource = tk.StringVar()
Sourcechoosen = ttk.Combobox( width = 27, textvariable = inpSource,state='readonly')
Sourcechoosen['values'] = ('Banglore', 
                        'Kolkata',
                        'Delhi',
                        'Chennai',
                        'Mumbai',)

Sourcechoosen.grid(row=3,column=1,sticky=tk.EW)
Sourcechoosen.current(0) 


label = tk.Label(text='Total_Stops',font=26).grid(row=3,column=2,sticky=tk.EW,padx=10)
inpStop = tk.StringVar()
Stopchoosen = ttk.Combobox( width = 27, textvariable = inpStop,state='readonly')
Stopchoosen['values'] = ('non-stop', 
                        '1 stop',
                        '2 stops',
                        '3 stops',
                        '4 stops',)

Stopchoosen.grid(row=3,column=3,sticky=tk.EW)
Stopchoosen.current(0) 
# text = tk.Entry().grid(row=2,column=3,sticky=tk.EW)

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
Deschoosen.current(0) 


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
Infochoosen.current(0) 
# text = tk.Entry().grid(row=3,column=3,sticky=tk.EW)

def saveinput():
    inp.append(inpAir.get())
    inp.append(inphour.get()+":"+inpmin.get())
    inp.append(cal.get_date().strftime("%m/%d/%Y"))
    inp.append(inp_h.get()+"h"+inp_m.get()+"m")
    inp.append(inpSource.get())
    inp.append(inpStop.get())
    inp.append(inpDes.get())
    inp.append(inpInfo.get())
    for i in inp:
        print(i)

btn = tk.Button(text="Save",width=10,command=saveinput).grid(row=5,column=0,columnspan=5,pady=120)
root.mainloop()