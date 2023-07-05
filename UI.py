import PySimpleGUI as sg
from tkinter import filedialog as fd

WIN_DIMENSIONS = (100,50)
VERSION = "V.4.0"
#Home window
def loadUI():
    img_dir = ''
    csv_file = ''
    #displays the following features
    layout = [[sg.Text("Please select an option:")], 
            [sg.Button("Generate Predictions")], 
            [sg.Button("Retrain System")]]
    
    # Create the window with a title
    window = sg.Window(title = f"MIQPC23 {VERSION}: Home", layout = layout, margins = WIN_DIMENSIONS)

    # Create an event loop
    retrain_model = None  #None: we clicked [X]; True: we clicked "Retrain System"; False: we clicked "Generate Predictions"
    while True:
        event, values = window.read()
        #if the window is closed, stop the program
        if event == sg.WIN_CLOSED:
            break
        #if we chose generate predictions, load the next window (browse files UI)
        elif event == "Generate Predictions":
            retrain_model = False
            window.close()
            img_dir = get_img_dir()
            break
        #if we chose retrain system, load the next window (retrain AI)
        elif event == "Retrain System":
            retrain_model = True
            window.close()
            img_dir = get_img_dir()
            csv_file = get_csv_file()
            break
    window.close() #closes the window
    return(retrain_model, img_dir, csv_file)
    

#UI for generating predictions
def get_img_dir():
    img_dir = "" #path to images
    layout = [[sg.Text("Please choose the folder with photos to analyze:")], 
            [sg.Button("Browse Folders")], 
            [sg.Button("CANCEL")]]

    #Create the window
    window = sg.Window(f"MIQPC23 {VERSION}: Select a directory", layout)

    #Create an event loop
    goBack = False     #did we click cancel?
    while True:
        event, values = window.read()
        #if press [X], close the window
        if  event == sg.WIN_CLOSED:
            break
        #if we click cancel, go back to the home window
        elif event == "CANCEL":
            goBack = True
            break
        #if we click "Browse Folders", set pathname to the selected folder    
        elif event == "Browse Folders":
            img_dir = fd.askdirectory()
            break
    window.close() #closes the window
    #if goBack is true, go back to loadUI()
    if(goBack):
        return loadUI()
    return img_dir

#UI for getting csv file with classes and image names. Only called for retraining purposes
def get_csv_file():
    csv_file = ""
    layout = [[sg.Text("Please select a CSV file that contains the image names and labels of its contents:")], 
            [sg.Button("Browse Files")], 
            [sg.Button("CANCEL")]]
    
    #Create the window
    window = sg.Window(f"MIQPC23 {VERSION}: CSV Selection", layout)

    #Create an event loop
    goBack = False
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if  event == sg.WIN_CLOSED:
            break
        elif event == "CANCEL":
            goBack = True
            break
        elif event == "Browse Files":
            csv_file = fd.askopenfilename() #TODO: verify that this returns JUST the file name, NOT the opened file
            break
    window.close()
    if(goBack):
        return loadUI()
    return csv_file