from tkinter import *
import tkinter as tk
import os
import sys
import time

win = tk.Tk()

def evaluationAgentRamdom():
    os.system("python3 eval_trained_model_5random.py " + var.get())


def evaluationFixedSystem():
    os.system("python3 eval_fixed_system_5random.py")


win.title('Demo Traffict Light Controll')
win.geometry("300x500+600+300")


var = tk.StringVar()
set1 = tk.OptionMenu(win,var,"RANDOM_1","RANDOM_2","RANDOM_3","RANDOM_4","RANDOM_5")
set1.config(font=("Arial",13))
set1.pack()
# set1.grid(row=1,column=0)


tk.Button(
        win, text="EVALUATE STL", font="Helvetica", command=evaluationFixedSystem,
        highlightbackground ="#8EF0F7", pady=2
    ).pack()



theLabel = Label(win,text="EVALUATION ANGENT 2:")
theLabel.pack()


tk.Button(
        win, text="EVALUATE AGENT 2", font="Helvetica", command=evaluationAgentRamdom,
        highlightbackground ="#8EF0F7", pady=2
    ).pack()


# https://www.youtube.com/watch?v=IB6VkXJVf0Y&list=PL6gx4Cwl9DGBwibXFtPtflztSNPGuIB_d&index=12 :Messsage Box


win.mainloop()