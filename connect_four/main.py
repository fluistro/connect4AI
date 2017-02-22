import kivy
from kivy.config import Config
Config.set('graphics','resizable',0)
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.modules import inspector
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import *
from kivy.properties import NumericProperty,ListProperty

global connectFourGame

class ConnectFour(Widget):
    board = ListProperty([[0]*6 for x in range(7)])
    def get_first_space(col):
        """
        Returns the index of the first space in a list that is 0
        """
        for i in range(len(col)):
            if col[i] == 0:
                return i
        return False

    def make_move(self,col_no):
        print("make_move: {}".format(col_no))
        self.board[col_no][ConnectFour.get_first_space(
            self.board[col_no])] = 1
        print("Board after move: {}".format(self.board))
    pass

class Column(Widget):
    col_no = NumericProperty(None)
    def on_touch_down(self,touch):
        global connectFourGame
        if self.collide_point(touch.x,touch.y):
            print("Move on Column: {}".format(self.col_no))
            connectFourGame.make_move(self.col_no)

class ConnectFourApp(App):
    def build(self):
        global connectFourGame
        connectFourGame = ConnectFour()
        inspector.create_inspector(Window,connectFourGame)
        Window.size=(800,600)
        return connectFourGame

def main():
    ConnectFourApp().run()
