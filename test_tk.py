import tkinter as tk

root = tk.Tk()
root.title("Tkinter Test Window")
root.geometry("300x200")
tk.Label(root, text="If you can see this, Tkinter works!").pack()
root.mainloop()