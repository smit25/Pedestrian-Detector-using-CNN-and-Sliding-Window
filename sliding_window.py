# IMPLEMENTATION OF SLIDING WINDOW


import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image,ImageDraw
import torch
import torchvision
import numpy as np
# from keras import backend as K
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform


"""
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        global Net_model 
        Net_model= load_model('test_save2.h5')

Net_model.summary()
print("smit")
# Net_model.evaluate()

"""
# LOADING THE MODEL
working_model=torch.load('model.pt')
working_model.eval()

# loading and pre-processing the image
def generate_test_img():
	img = Image.open('test4.jpg')
	img=img.convert('L')
	print(img.size)
	# img.save('rgb.jpg')
	if img.size[0]>2000 and img.size[1]>1000:
		img=img.resize((1000,2000),Image.ANTIALIAS)
	else:
		img=img.resize((img.size[0],img.size[1]),Image.ANTIALIAS)
	print(img.mode)
	return img

#converting the image to numpy array
def generate_test_img_array(img):
	h=img.size[0]
	w=img.size[1]
	img=np.array(img)
	img=img/255
	img=img.reshape(1,1,h,w).astype('float32')
	print(img.shape)
	return img

# calling the function for inputting the test image
test1=generate_test_img()
test1_arr=generate_test_img_array(test1)


def sw(image,image_array, x =1.6, y=2.4, save_files = True, threshold = 0.97):


	#CONVERTING NUMPY ARRAY TO TENSOR AND PUTTING IT IN MODEL
	image_tensor=torch.from_numpy(image_array)
	output_tensor = working_model(image_tensor)
	print("out",output_tensor.shape)
	output_squeeze = torch.squeeze(output_tensor,0).detach().numpy() # all dimensions of size 1 removed
	print(output_squeeze.shape)

	heatmap = output_squeeze[0] # the channel or the 2nd one is eliminated
	print(heatmap,"HEATMAP",heatmap.shape)
	heatmap_thr = heatmap.copy()

	# applying threshold or a simpler version of non-max suppression
	heatmap_thr[heatmap[:,:]>threshold] = 100
	heatmap_thr[heatmap[:,:]<=threshold] = 0

	boxes=[]

	#converting PIL image to DrawImage
	image_copy = image.copy()
	draw = ImageDraw.Draw(image)
	draw_copy = ImageDraw.Draw(image_copy)

	heatmap_img = Image.fromarray(np.uint8(cm.gist_heat(heatmap)*255))
	heatmap_img.show()

	print("with threshold")
	heatmap_img_thresh = Image.fromarray(np.uint8(cm.gist_heat(heatmap_thr)*255))
	heatmap_img_thresh.show()
	# print(heatmap.shape[0])
	print("---------")
	# print(np.arange(heatmap.shape[0]))
	print("----------")
	
	# yy-rows have the same elements, xx- columns have the same elements
	xx,yy = np.meshgrid(np.arange(heatmap.shape[1]),np.arange(heatmap.shape[0])) 
	
	# x_det and y_det contain all those indices of heatmap whose value is greater than threshold.
	x_det = (xx[heatmap[:,:]>threshold]) 
	y_det = (yy[heatmap[:,:]>threshold])
	print(y_det)

	# Scaling of the model
	shrink_ratio = (image.width/heatmap_img.width , image.height/heatmap_img.height)
	
	# Appending the dimensions of the bounding boxes
	for i,j in zip(x_det,y_det): 
		if not save_files :
			if i>heatmap_img.width//1.95 and j>int(heatmap_img.height/1.95) :
				boxes.append([int(i*13),int(j*13),int(48),int(96)])
		else :
			boxes.append([int(i*13),int(j*13),int(48),int(96)])

	#group interecting rectanlgles
	bound_boxes = cv2.groupRectangles(boxes,2,1)
	bound_boxes=bound_boxes[:1]

	print("Number of Objects: ",len(bound_boxes[0]))
	print(bound_boxes)

	# draw the box on the image and its copy
	for box in bound_boxes:
		for b in box:
			print(b,"BOX")
			draw_copy.rectangle((b[0],b[1],b[2]+b[0],b[1]+b[3]),outline='blue')#draw.draw_rectangle(xy, ink, 0, width)

	for box in bound_boxes[0]:
		draw.rectangle((box[0]-(x-1)*box[2]//2,box[1]-(y-1)*box[2]//2,box[2]*x+box[0]-(x-1)*box[2]//2,box[3]*y+box[1]-(y-1)*box[2]//2),outline='green')

	#saving the images
	if(save_files) :
		image.save("Actual output.png")
		image_copy.save("Copy_Output.png")
		heatmap_img.save("Heatmap.png")
		heatmap_img_thresh.save("Heatmap_thresh.png")
	
	image.show()

# calling the function
sw(test1,test1_arr)