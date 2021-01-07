class WaveForm:
    def __init__(self, blw):
        self.blw=blw
        self.peaks=[]
        self.hits=[]
        self.init=0

class Hit:
    def __init__(self, init, fin):
        self.init=init
        self.fin=fin
        self.groups=[]
        self.area=0
        self.height=0

class Group:
    def __init__(self, maxi, left, right, height):
        self.maxi=maxi
        self.left=left
        self.right=right
        self.height=height
