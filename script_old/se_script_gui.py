# Etienne.St-Onge@usherbrooke.ca

# TODO (in hold for now)
"""
import sys
from PyQt4 import QtGui
import argparse
import argparseui

app = QtGui.QApplication(sys.argv)
a = argparseui.ArgparseUi(lps_parser)
a.show()
app.exec_()
"""

"""
import os
import Tkinter as tk
from se_script_config import bool_dict, var_dict, params_dict, ordered_key


root = tk.Tk()
var = tk.IntVar()
label = tk.Label(root)
def sel():
    selection = "You selected the option " + str(var.get())
    label.config(text = selection)

def main():
    
    #test
    i = 0
    j = 0
    for key in ordered_key:
        label = tk.Label(root, text=key + " :")
        label.grid(row=j,column=0)
        j += 1
        for bool_param in bool_dict[key]:
            if bool_param[1] is None:
                r1 = tk.Radiobutton(root, text="None", variable=var, value=i, command=sel)
            else:
                r1 = tk.Radiobutton(root, text='"' + bool_param[1] + '"' , variable=var, value=i, command=sel)
            r1.grid(row=j,column=1)
            if bool_param[2] is None:
                r2 = tk.Radiobutton(root, text="None", variable=var, value=i, command=sel)
            else:
                r2 = tk.Radiobutton(root, text='"' + bool_param[2] + '"' , variable=var, value=i, command=sel)
            r2.grid(row=j,column=2)
            
            j += 1 
            
            
            
        i += 1
    
    root.mainloop()
        

           

if __name__ == '__main__':
    main()
"""