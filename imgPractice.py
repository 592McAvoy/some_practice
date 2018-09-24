from PIL import Image
import numpy as np

class BiLinerInterpolation():
    def __init__(self,img):
        self.img = img #shape (H,W,C)

    def deal(self, h, w):
        #print ("h:%f w%f\n"%(h,w))
        H,W,C = self.img.shape
        w1 = int(np.floor(w))
        w2 = int(min(w1 + 1,W-1))
        h1 = int(np.floor(h))
        h2 = int(min(h1 + 1,H-1))
        #print ("w1:%d w2:%d h1:%d h2:%d\n"%(w1,w2,h1,h2) )

        a11 = self.img[h1,w1,:]
        a12 = self.img[h1,w2,:]
        a21 = self.img[h2,w1,:]
        a22 = self.img[h2,h2,:]

        x1 = (w2-w)/1*a11 + (w-w1)/1*a12
        x2 = (w2-w)/1*a21 + (w-w1)/1*a22

        target = (h2-h)/1*x1 + (h-h1)/1*x2

        return target

class BiCubicInterpolation():
    def __init__(self,img):
        self.img = img #shape (H,W,C)

    def biCubic(self, x, a=-0.5):
        x1 = abs(x)
        x2 = abs(x) ** 2
        x3 = abs(x) ** 3
        if(x1 <= 1):
            return (a+2)*x3 - (a+3)*x2 + 1
        if(x1 < 2):
            return a*x3 - 5*a*x2 + 8*a*x1 - 4*a
        else:
            return 0

    def deal(self, h, w):
        #print ("h:%f w%f\n"%(h,w))
        H,W,C = self.img.shape

        w_start = int(max(np.floor(w)-1,0))
        w_end = int(w_start + 4)
        h_start = int(max(np.floor(h)-1,0))
        h_end = int(h_start + 4)
        if(h_end > H):
            h_end = H
            h_start = h_end - 4
        if(w_end > W):
            w_end = W
            w_start = w_end - 4
       
        target = np.zeros(C)

        for hh in range(h_start,h_end):
            for ww in range(w_start,w_end):
                w_h = self.biCubic(hh-h)
                w_w = self.biCubic(ww-w)
                target += w_h*w_w*self.img[hh,ww,:]
        
        return target


class handleImg():
    def __init__(self, img, scale=3):
        self.img = img
        self.scale = scale
    
    def do(self, intp):
        H,W,C = self.img.shape
        newW = W * self.scale
        newH = H * self.scale
        result = np.zeros((newW, newH, C))

        print("new shape:")
        print(result.shape)

        for h in range(newW):
            for w in range(newH):
                result[h,w,:] = intp.deal(h/self.scale,w/self.scale)
        return result



if __name__ == '__main__': 
    im = Image.open('test.png')
    #im.show()

    img = np.array(im)
    print("shape:")
    print(img.shape)

    #interp = BiLinerInterpolation(img)
    interp = BiCubicInterpolation(img)

    handler = handleImg(img,2)
    result = handler.do(interp)
    print("final shape:")
    print(result.shape)

    newImg = Image.fromarray(np.uint8(result))
    #newImg.show()
    #newImg.save('max2in2.png')
    newImg.save('max2in3.png')

                


