from tkinter import *
import tkinter as tk

win = Tk()

var = tk.StringVar()
set1 = tk.OptionMenu(win,var,"DO","Le")
set1.config(font=("Arial",25))
set1.grid(row=1,column=0)

win.mainloop()