import cv2
from os import path

# for i in range(74):
# 	num = str(i)
# 	load_path = 'PixelLabelData/' +  'Label_' + num + '.png';
# 	if(path.exists(load_path)):
# 		img = cv2.imread(load_path, -1);
# 		img_resized = cv2.resize(img, (1080, 1080)); 
# 		save_path = 'rescaled-PixelLabelData/' +  'Label_' + num + '.png';
# 		cv2.imwrite(save_path,img_resized)


# j = 1
# for i in range(6632):
# 	num = str(i)
# 	load_path = 'LVAO-gray/' + num + '-LVAO.jpg';
# 	if(path.exists(load_path)):
# 		img = cv2.imread(load_path, -1);
# 		img_resized = cv2.resize(img, (1080, 1080));
# 		if j < 10: 
# 			save_path = 'frames/frame_00' + str(j) + '.png';
# 		else:
# 			save_path = 'frames/frame_0' + str(j) + '.png';
# 		cv2.imwrite(save_path,img_resized)
# 		j = j+1


for i in range(74):
	num = str(i)
	load_path = ''
	if i < 10:
		load_path = 'masks_name/mask_00' + num + '.png'
	else:
		load_path = 'masks_name/mask_0' + num + '.png'
	if(path.exists(load_path)):
		img = cv2.imread(load_path, 0)
		save_path = ''
		if i < 10:
			save_path = 'masks/frame_00' + num + '.png'
		else:
			save_path = 'masks/frame_0' + num + '.png'
		cv2.imwrite(save_path, img)

