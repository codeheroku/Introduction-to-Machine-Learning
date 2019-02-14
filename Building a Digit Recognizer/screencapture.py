import pyscreenshot as ImageGrab
import time

images_folder = "orig_images/5/"

for i in range (0,45):
	
	time.sleep(5)
	im = ImageGrab.grab(bbox=(80, 80, 208, 208)) # X1,Y1,X2,Y2
	print "saved....",i
	im.save(images_folder+str(i)+'.png')
	print "clear screen now and redraw now..."
	