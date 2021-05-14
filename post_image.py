from PIL import Image
import numpy as np
img_ketton = Image.open("ketton.tif").convert("1")
mask_ketton=np.where(np.array(img_ketton),0,1)
mask_ketton=mask_ketton.T
# mask_ketton=np.zeros(shape=(1024,512))
print(mask_ketton)
np.savetxt("ketton.txt",mask_ketton) 
# img_ketton.show()

# img_beadpack = Image.open("beadpack.tif").convert("1")
# mask_beadpack=np.where(np.array(img_beadpack),0,1)
# print(mask_beadpack)
# np.savetxt("beadpack.txt",mask_beadpack) 
# # img_beadpack.show()

# img_berea = Image.open("berea.tif").convert("1")
# mask_berea=np.where(np.array(img_berea),1,0)
# print(mask_berea)
# np.savetxt("berea.txt",mask_berea) 
# # img_berea.show()

# img_fc = Image.open("fc.bmp").convert("L")
# mask_fc=np.where(np.array(img_fc),0,1)
# # mask_fc=mask_fc.T
# print(mask_fc.shape)
# np.savetxt("fc.txt",mask_fc) 
# # Image.fromarray(img_fc).show()

img_xct_l = Image.open("xct_0414_left.tif")
mask_xct_l=np.where(np.array(img_xct_l)>150,0,1)
# mask_xct_l=np.zeros_like(mask_xct_l)
print(mask_xct_l.shape)
print(mask_xct_l)
np.savetxt("xct_0414_left.txt",mask_xct_l) 

img_xct_r = Image.open("xct_0414_right.tif")
mask_xct_r=np.where(np.array(img_xct_r)>150,0,1)
# mask_xct_r=np.zeros_like(mask_xct_r)
print(mask_xct_r.shape)
print(mask_xct_r)
np.savetxt("xct_0414_right.txt",mask_xct_r) 