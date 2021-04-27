#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__
#include <gl/freeglut.h>

class CPUBitmap {
private:
	unsigned char* pixels;
	int x, y;
	void* dataBlock;
	void (*bitmapExit)(void*);
	static CPUBitmap** get_bitmap_ptr(void) {
		static CPUBitmap* gBitmap;
		return &gBitmap;
	}
	static void Key(unsigned char key, int x, int y) {
		switch (key) {
		case 27:
			CPUBitmap * bitmap = *(get_bitmap_ptr());
			if (bitmap->dataBlock != nullptr && bitmap->bitmapExit != nullptr) {
				bitmap->bitmapExit(bitmap->dataBlock);
			}
			exit(0);
		}
	}
	static void Draw(void) {
		CPUBitmap* bitmap = *(get_bitmap_ptr());
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
		glFlush();
	}
public:
	CPUBitmap(int width, int height, void* d = nullptr) {
		pixels = new unsigned char[width * height * 4];
		x = width;
		y = height;
		dataBlock = d;
	}
	~CPUBitmap() {
		delete[] pixels;
	}
	unsigned char* get_ptr(void) const { return pixels; }
	long image_size(void) const { return x * y * 4; }
	void display_and_exit(void(*e)(void*) = nullptr) {
		CPUBitmap** bitmap = get_bitmap_ptr();
		*bitmap = this;
		bitmapExit = e;
		int c = 1;
		char* dummy = "";
		glutInit(&c, &dummy);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
		glutInitWindowSize(x, y);
		glutCreateWindow("bitmap");
		glutKeyboardFunc(Key);
		glutDisplayFunc(Draw);
		glutMainLoop();
	}
	
};

#endif