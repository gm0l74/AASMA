#!/usr/bin/env python3
#---------------------------------
# AASMA - Environment
# File : about_window.py
#
# @ start date          25 04 2020
# @ last update         25 04 2020
#---------------------------------

#---------------------------------
# Imports
#---------------------------------
import tkinter as tk

#---------------------------------
# class Application
#---------------------------------
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("AASMA GOD'S EYE")
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")
