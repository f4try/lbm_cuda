import taichi as ti
ti.init(arch=ti.cpu)
@ti.data_oriented
class add_test:
    def __init__(self):
        self.N=1<<20
        self.x = ti.field(shape=self.N,dtype=ti.f32)
        self.y = ti.field(shape=self.N,dtype=ti.f32)
    @ti.kernel
    def init(self):
        for i in ti.ndrange(self.N):
            self.x[i]=1.
            self.y[i]=2.
    @ti.kernel
    def add(self):
        for i in ti.ndrange(self.N):
            self.y[i]+=self.x[i]
    
    def solve(self):
        self.init()
        self.add()

add_t = add_test()
add_t.solve()
print(add_t.y[0])
    