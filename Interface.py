from NeuralNetClass import NeuralNet
from tkinter import *
import tkinter.messagebox


class ProgramInterface(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        # Setup frame
        container = Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        frame = MainMenu(container, self)
        self.frames[MainMenu] = frame
        frame.grid(row=0, column=0, stick="nesw")

        self.show_frame(MainMenu)

    def show_frame(self, context):
        frame = self.frames[context]
        frame.tkraise()


class MainMenu(Frame):
    def __init__(self, parent, controller):
        self.NeuralNetwork = NeuralNet()
        self.trainedFirst = False
        Frame.__init__(self, parent)

        intro = Label(self, text="Digit Recognition using a Neural Network")
        intro.grid(row=0, column=1, padx=10, pady=10)

        button1 = Button(self, text="TRAIN", fg="red")
        button1.bind("<Button-1>", self.TrainNeuralNetwork)
        button1.grid(row=1, column=1, ipadx=10, ipady=10)
        button1Explanation = Label(self, text="Train the Neural Network \nusing 2,500 images")
        button1Explanation.grid(row=1, column=2, padx=10, pady=10, sticky=W)

        button2 = Button(self, text="TEST", fg="red")
        button2.bind("<Button-1>", self.TestNeuralNetwork)
        button2.grid(row=2, column=1, pady=10, ipadx=10, ipady=10)
        button2Explanation = Label(self, text="Test the Neural Network \nusing 9,000 images")
        button2Explanation.grid(row=2, column=2, padx=10, pady=10, sticky=W)

        button3 = Button(self, text="RANDOM TEST", fg="red")
        button3.bind("<Button-1>", self.TestCustomNeuralNetwork)
        button3.grid(row=3, column=1, pady=10, ipadx=10, ipady=10)
        button3Explanation = Label(self, text="Test the Neural Network \nusing a chosen images")
        button3Explanation.grid(row=3, column=2, padx=10, pady=10, sticky=W)

        bottomText = Label(self, text="By Ian Hamilton")
        bottomText.grid(row=4, column=1, pady=10)

    def TrainNeuralNetwork(self, event):
        answer = tkinter.messagebox.askokcancel("Warning", "This operation will take a very long time.\nProceed?")

        if answer == 1:  # Clicked OK
            print("Clicked Train")
            # Run Training Neural Net
            self.NeuralNetwork.trainNeuralNetwork()
            self.trainedFirst = True
            tkinter.messagebox.showinfo("Success", "Training Complete")

    def TestNeuralNetwork(self, event):
        if self.trainedFirst:
            print("Clicked Test")
            testResults = self.NeuralNetwork.testNeuralNetwork()
            numTestingImages = self.NeuralNetwork.getNumberofTestingImages()
            numTrainingImages = self.NeuralNetwork.getNumberofTrainingImages()
            numTrainingSteps = self.NeuralNetwork.getNumberofTrainingSteps()

            tkinter.messagebox.showinfo("Test Results", ("Number of Training Images: " + str(numTrainingImages) + "\n"
                                                         + "Number of Training Steps: " + str(numTrainingSteps) + "\n\n"
                                                         + "Number of Testing Images: " + str(numTestingImages) + "\n\n"
                                                         + "Accuracy: " + str(testResults) + "%"))
        else:
            tkinter.messagebox.showinfo("Error", "You need to train the Neural Network first")


    def TestCustomNeuralNetwork(self, event):
        if self.trainedFirst:
            testImageIndex, predictedValue, actualValue,  = self.NeuralNetwork.testSingularNeuralNetwork()
            print("Clicked Custom Test")

            tkinter.messagebox.showinfo("Test Result", ("Predicted Result: " + str(predictedValue)))

            # To preview the images
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt

            plt.show(plt.imshow(self.NeuralNetwork.testingImages[testImageIndex][0]))

        else:
            tkinter.messagebox.showinfo("Error", "You need to train the Neural Network first")


Interface = ProgramInterface()
Interface.mainloop()
