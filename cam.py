from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
from PIL import ImageTk, Image
import cv2
import numpy as np
import tensorflow as tf
loaded_model = tf.keras.models.load_model('model.h5')


#Login Function 
def clear():
    userentry.delete(0,END)
    passentry.delete(0,END)

def close():
    win.destroy()

def login():
    if user_name.get()=="" or password.get()=="":
        messagebox.showerror("Error","Enter User Name And Password",parent=win)	
    else:
        try:
            con =  mysql.connector.connect(host="localhost",database='python_project', user="root", password="admin")
            cur = con.cursor()
            cur.execute("select * from details where U_Name=%s and Passw = %s",(user_name.get(),password.get()))
            row = cur.fetchone()
            if row==None:
                messagebox.showerror("Error" , "Invalid User Name And Password", parent = win)

            else:
                messagebox.showinfo("Success" , "Successfully Login" , parent = win)
                close()
                face_emotion()
            con.close()
        except Exception as es:
            messagebox.showerror("Error", parent = win)


# face_emotion Panel 
cap = None
def face_emotion():

    root = Tk()
    # Create a frame
    app = Frame(root, bg="white")
    app.grid()
    # Create a label in the frame

    lmain = Label(app)
    # Create text widget and specify size.
    T = Text(root, height = 5, width = 52)

    # Create label
    l = Label(root, text = "Fact of the Day")
    l.config(font =("Courier", 14))

    Fact = "The emotion will be displayed here"
    T.insert("1.0",Fact)
    T.grid()
    T.place(anchor = NW)

    def predict():
        newsize = (48, 48)
        test_img = face[-1].resize(newsize)
        test_img = test_img.convert('LA')
        # getting the pixel values from the image
        tuple_data = list(test_img.getdata())

        data = []
        for tuple in tuple_data:
            data.append(tuple[0])

        #reshaping the pixel values to make it compatible with the model
        pixel_matrix = np.array(data).reshape(1, 48, 48, 1)
        pixel_matrix = pixel_matrix/255
        print(pixel_matrix.shape)
        yhat_test = loaded_model.predict_classes(pixel_matrix)
        # displaying the output on the webpage
        if (yhat_test==0):
            return'The person in this image appears to be angry.'
        elif (yhat_test==1):
            return 'The person in this image appears to be happy.'

        elif (yhat_test==2):
            return 'The person in this image appears to be sad.'

        else: # yhat_test==3
            return 'The person in this image appears to be surprised.'
    def emotion():
        #root.geometry("500x500")

        root.title("emotion")
        root.configure(background='black')
        T.delete("1.0",END)
        Emotion = predict()
        T.insert("1.0", Emotion)
    b = Button(root, text = "Get emotion", command=emotion)
    b.place(anchor = CENTER)
    b.grid()
    lmain.grid()
    
    global cap
    # Capture from camera
    cap = cv2.VideoCapture(0)

    # function for video streaming
    face = []
    i = 0

    def video_stream():

        _, frame = cap.read()

        frame=cv2.flip(frame,1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        img = Image.fromarray(cv2image)
        face.append(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, video_stream)


    video_stream()
    root.mainloop()
    cap.release()
    cv2.destroyAllWindows() 


# Signup Window 

def signup():
    # signup database connect 
    def action():
        if user_name.get()=="" or password.get()=="" or very_pass.get()=="":
            messagebox.showerror("Error" , "All Fields Are Required" , parent = primary_frame)
        elif password.get() != very_pass.get():
            messagebox.showerror("Error" , "Password & Confirm Password Should Be Same" , parent = primary_frame)
        else:
            try:
                con= mysql.connector.connect(host="localhost",database='python_project', user="root", password="admin")
                cur = con.cursor()
                print(user_name.get())
                t="select * from details where U_Name = '" + user_name.get() +"'"
                cur.execute(t)
                print(user_name.get())
                row = cur.fetchone()
                if row!=None:
                    messagebox.showerror("Error" , "User Name Already Exits", parent = primary_frame)
                else:
                    cur.execute("insert into details(U_Name,Passw) values(%s,%s)",
                        (
                        user_name.get(),
                        password.get()
                        ))
                    con.commit()
                    con.close()
                    messagebox.showinfo("Success" , "Ragistration Successfull" , parent = primary_frame)
                    clear()
                    switch()

            except Exception as es:
                messagebox.showerror("Error", parent = primary_frame)

    # close signup function			
    def switch():
        primary_frame.destroy()

    # clear data function
    def clear():
        user_name.delete(0,END)
        password.delete(0,END)
        very_pass.delete(0,END)


    # start Signup Window	

    primary_frame = Tk()
    primary_frame.title("signup")
    primary_frame.maxsize(width=500 ,  height=600)
    primary_frame.minsize(width=500 ,  height=600)
    primary_frame.configure(background='black')

    #heading label
    heading = Label(primary_frame , text = "Signup" , font = 'Verdana 30 bold')
    heading.place(x=80 , y=60)

    # form data label
    user_name = Label(primary_frame, text= "User Name :" , font='Verdana 10 bold')
    user_name.place(x=80,y=130)

    password = Label(primary_frame, text= "Password :" , font='Verdana 10 bold')
    password.place(x=80,y=160)

    very_pass = Label(primary_frame, text= "Verify Password:" , font='Verdana 10 bold')
    very_pass.place(x=80,y=190)

    # Enter Box

    user_name = StringVar()
    password = StringVar()
    very_pass = StringVar()

    user_name = Entry(primary_frame, width=40,textvariable = user_name)
    user_name.place(x=200 , y=130)

    password = Entry(primary_frame, width=40,show="*" ,textvariable = password)
    password.place(x=200 , y=160)

    very_pass= Entry(primary_frame, width=40,show="*" , textvariable = very_pass)
    very_pass.place(x=201 , y=190)

    # button login and clear

    btn_signup = Button(primary_frame, text = "Signup" ,font='Verdana 10 bold', command = action)
    btn_signup.place(x=200, y=413)


    btn_login = Button(primary_frame, text = "Clear" ,font='Verdana 10 bold' , command = clear)
    btn_login.place(x=280, y=413)


    sign_up_btn = Button(primary_frame , text="Switch To Login" , command = switch )
    sign_up_btn.place(x=350 , y =20)


    primary_frame.mainloop()

# Login Window 

win = Tk()

# app title
win.title("login")

# window size
win.maxsize(width=500 ,  height=500)
win.minsize(width=500 ,  height=500)
win.configure(background='black')



#heading label
heading = Label(win , text = "Login" , font = 'Verdana 25 bold')
heading.place(x=80 , y=150)

username = Label(win, text= "User Name :" , font='Verdana 10 bold')
username.place(x=80,y=220)

userpass = Label(win, text= "Password :" , font='Verdana 10 bold')
userpass.place(x=80,y=260)

# Entry Box
user_name = StringVar()
password = StringVar()

userentry = Entry(win, width=40 , textvariable = user_name)
userentry.focus()
userentry.place(x=200 , y=223)

passentry = Entry(win, width=40, show="*" ,textvariable = password)
passentry.place(x=200 , y=260)


# button login and clear

btn_login = Button(win, text = "Login" ,font='Verdana 10 bold',command = login)
btn_login.place(x=200, y=293)


btn_login = Button(win, text = "Clear" ,font='Verdana 10 bold', command = clear)
btn_login.place(x=260, y=293)

# signup button

sign_up_btn = Button(win , text="Switch To Sign up" , command = signup )
sign_up_btn.place(x=350 , y =20)

win.mainloop()
try:
    cap.release()
    cv2.destroyAllWindows()
except:
    cv2.destroyAllWindows()
