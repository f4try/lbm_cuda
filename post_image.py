from PIL import Image
import numpy as np
img_ketton = Image.open("ketton.tif").convert("1")
mask_ketton=np.where(np.array(img_ketton),0,1)
mask_ketton=mask_ketton.T
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

img_fc = Image.open("fc.bmp").convert("L")
mask_fc=np.where(np.array(img_fc),0,1)
# mask_fc=mask_fc.T
print(mask_fc.shape)
np.savetxt("fc.txt",mask_fc) 
# Image.fromarray(img_fc).show()