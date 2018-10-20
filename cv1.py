from PIL import Image
import numpy as np

class BiImage():
    def __init__(self, img, SE):
        H,W,C = img.shape
        self.img = img.reshape(C,H,W)
        self.SE = SE

    def setCenter(self, row, col):
        self.centR = row
        self.centC = col

    def toBiImage(self):
        C,H,W = self.img.shape
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if self.img[c,h,w]>128:
                        self.img[c,h,w]=255
                    else:
                        self.img[c,h,w]=0
        return self.img.reshape(H,W,C)
                        

    def dilation(self):
        result = self.img.copy()
        print(result[0,20:22,:10])

        C,H,W = result.shape
        HH, WW = self.SE.shape
        print(self.SE)
    
        for c in range(C):
            for h in range(H-HH+1):
                for w in range(W-WW+1):
                    area = self.img[c,h:h+HH,w:w+WW]
                    
                    match = False
                    for hh in range(HH):
                        for ww in range(WW):
                            if self.SE[hh,ww] == 255 and area[hh,ww] == self.SE[hh,ww]:
                                match = True
                                break

                    if match:        
                        result[c,h+1,w+1] = 255
                    else:
                        result[c,h+1,w+1] = 0
                    
        print(result[0,20:22,:10])           
        result = result.reshape(H,W,C)
        return result
    
    def erosion(self):
        result = self.img.copy()
        print(result[0,20:22,10:20])

        C,H,W = result.shape
        HH, WW = self.SE.shape
        print(self.SE)
    
        for c in range(C):
            for h in range(H-HH+1):
                for w in range(W-WW+1):
                    area = self.img[c,h:h+HH,w:w+WW]
                    
                    match = True
                    for hh in range(HH):
                        for ww in range(WW):
                            if self.SE[hh,ww] == 255 and area[hh,ww] != self.SE[hh,ww]:
                                match = False
                                break

                    if match:        
                        result[c,h+1,w+1] = 255
                    else:
                        result[c,h+1,w+1] = 0
                    
        print(result[0,20:22,10:20])           
        result = result.reshape(H,W,C)
        return result


class GrayImage():
    def __init__(self, img, SE):
        H,W,C = img.shape
        self.img = img.reshape(C,H,W)
        self.SE = SE

    def setCenter(self, row, col):
        self.centR = row
        self.centC = col

    def dilation(self):
        result = self.img.copy()

        C,H,W = result.shape
        HH, WW = self.SE.shape
    
        for c in range(C):
            for h in range(H-HH+1):
                for w in range(W-WW+1):
                    area = self.img[c,h:h+HH,w:w+WW]
                    area = area + self.SE
                    val = np.max(area)
                    if val >= 256:
                        val = 255
                    if val < 0:
                        val = 0
                    result[c,h+1,w+1] = val
                   
        result = result.reshape(H,W,C)
        return result
    
    def erosion(self):
        result = self.img.copy()

        C,H,W = result.shape
        HH, WW = self.SE.shape
    
        for c in range(C):
            for h in range(H-HH+1):
                for w in range(W-WW+1):
                    area = self.img[c,h:h+HH,w:w+WW]
                    area = area - self.SE
                    val = np.min(area)
                    if val >= 256:
                        val = 255
                    if val < 0:
                        val = 0
                    result[c,h+1,w+1] = val
                   
        result = result.reshape(H,W,C)
        return result
    

if __name__ == '__main__':
    filename = 'test.png'

    im = Image.open(filename)
    
    img = np.array(im)
    SE = np.array([[255,255],[255,0]])

    #gimg = GrayImage(img,SE)
    bimg = BiImage(img,SE)
    bimg.toBiImage()
                  
    bimg.setCenter(1,1)

    #result = bimg.dilation()
    result = bimg.erosion()
    
    newImg = Image.fromarray(np.uint8(result))
    newImg.save('bi.png')

