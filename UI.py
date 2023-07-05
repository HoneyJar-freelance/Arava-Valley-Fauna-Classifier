import PySimpleGUI as sg
from tkinter import filedialog as fd

WIN_DIMENSIONS = (100,50) #dimensions of GUI
VERSION = "V.4.0" #version of software

#Home window
def loadGUI():
    '''
    Loads the main GUI.
    Returns: Tuple(Boolean|None, str, str)
    '''
    img_dir = ''
    csv_file = ''
    #displays the following features
    layout = [[sg.Text("Please select an option:")], 
            [sg.Button("Generate Predictions")], 
            [sg.Button("Retrain Model")]]
    
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
    
def get_img_dir():
    '''
    Prompts the user for a directory of images.
    Returns: str of a directory path
    '''
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
        return loadGUI()
    return img_dir

#UI for getting csv file with classes and image names. Only called for retraining purposes
def get_csv_file():
    '''
    Gets the file path to a CSV with important information for training.
    Returns: str of a file path.
    '''
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
            csv_file = fd.askopenfilename(filetypes=(('.csv')))
            break
    window.close()
    if(goBack):
        return loadGUI()
    return csv_file

def load_dependency_not_found_prompt():
    '''
    Informs the user that files are missing, and how to fix the issue.
    Returns: Tuple(str,str) | None
    '''

    train_files = None

    layout = [[sg.Text(f"Warning: important file(s) are missing.")],
              [sg.Text("You can either contact the developer or retrain a fresh model to generate all required files, or close the application.")],
              [sg.Button("Train new model")],
              [sg.Button("Exit")]]
    
    window = sg.Window(f"MIQPC23 {VERSION}: DependenciesNotFoundException", layout) #TODO: move this process to an exception handler
    
    #Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if  event == sg.WIN_CLOSED:
            break
        elif event == "Exit":
            break
        elif event == "Train new model":
            #Dont call other functions, as they will return to loadGUI
            train_files = (fd.askdirectory(),fd.askopenfilename(filetypes=(('.csv'))))
    window.close()

    return train_files

def give_error(msg:str):
    '''
    Notifies the user about an unknown error.
    '''
    layout = [[sg.Text(f"Unknown Error: Something went wrong.")],
              [sg.Text(msg)],
              [sg.Button("Exit")]]
    
    window = sg.Window(f"MIQPC23 {VERSION}: ERROR", layout) #TODO: move this process to an exception handler
    
    #Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if  event == sg.WIN_CLOSED:
            break
        elif event == "Exit":
            break
    window.close()
    