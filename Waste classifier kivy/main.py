from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from model import TensorFlowModel
# Import other dependencies
import cv2
import tensorflow as tf
import os
import numpy as np

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text='wating for picture to decide', size_hint=(1,.1))
        #self.button2 =  Button(text="wait", on_press=self.do_nothing, size_hint=(1,.1))
        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        #layout.add_widget(self.button2)
        # Load tensorflow/keras model
        
        
        #self.model = tf.keras.models.load_model("model_res152.h5")
        self.model = TensorFlowModel()
        self.model.load(os.path.join(os.getcwd(), 'model_152.tflite'))
        
        
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    def do_nothing(self,*args):
        pass
    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        #frame = frame[120:120+256, 200:200+256, :]
       
        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
     
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        # Return image
        return img_array

  
    def verify(self, *args):

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('input_image.jpg')
        ret, frame = self.capture.read()
        #frame = frame[120:120+256, 200:200+256, :]
        cv2.imwrite(SAVE_PATH, frame)

        
        classes =['Batteries', 'Clothes', 'E-waste', 'Glass', 'Light Blubs', 'Metal', 'Organic', 'Paper', 'Plastic']


        links={
            "Batteries": 'https://batteryuniversity.com/article/bu-705-how-to-recycle-batteries',
            "Clothes":  'https://www.treehugger.com/textile-recycling-5203438 ',
            "E-waste":  'https://www.conserve-energy-future.com/e-waste-recycling-process.php ',
            "Glass":  ' https://www.greenjournal.co.uk/2021/04/how-to-recycle-all-types-of-glass/ ',
            "Light Blubs":  ' https://www.treehugger.com/light-bulb-recycling-5206232 ',
            "Metal":  ' https://www.conserve-energy-future.com/recyclingmetal.php ',
            "Organic":  ' https://microbenotes.com/organic-waste-recycling/ ',
            "Paper":  ' https://www.cleanipedia.com/gb/sustainability/recycle-paper.html ',
            "Plastic":  ' https://www.conserve-energy-future.com/recyclingplastic.php ',
            }

        img_array = self.preprocess(SAVE_PATH)
        result = self.model.pred(img_array)
        #result = self.model.predict(img_array)
        index1 = np.argmax(result)
       
        print( classes[index1])
        self.verification_label.text=classes[index1]
        #self.button2.text="click to see how you can recycle"
        #self.button2.on_press=self.open_link(classes[index1])
        #self.verification_label.on_ref_press(webbrowser.open(links[classes[index1]]))
        #return classes[index1], data[classes[index1]]



if __name__ == '__main__':
    CamApp().run()