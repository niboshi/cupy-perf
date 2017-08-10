import cupy
import numpy

import cupy_perf


class Perf1(cupy_perf.PerfCases):
    def setUp(self):
        shape_tiny = (2, 3)
        shape_huge = (2000, 300)
        self.a = cupy.ones(shape_tiny, numpy.float32)
        self.b = cupy.ones(shape_tiny, numpy.float32)
        self.c = cupy.ones(shape_tiny, numpy.float32)
        self.a_huge = cupy.ones(shape_huge, numpy.float32)
        self.b_huge = cupy.ones(shape_huge, numpy.float32)
        self.c_huge = cupy.ones(shape_huge, numpy.float32)

    def perf_sum(self):
        cupy.sum(self.a)

    def perf_sum_huge(self):
        cupy.sum(self.a_huge)

    def perf_add(self):
        cupy.add(self.a, self.b)

    def perf_add_out(self):
        cupy.add(self.a, self.b, out=self.c)

    def perf_add_out_huge(self):
        cupy.add(self.a_huge, self.b_huge, out=self.c_huge)

    def perf_userkernel(self):
        kern = cupy.ElementwiseKernel(
            '''T a, T b''', '''T c''',
            '''c = a + b;''', 'test_kernel')
        kern(self.a, self.b, self.c)

    def perf_userkernel_huge(self):
        kern = cupy.ElementwiseKernel(
            '''T a, T b''', '''T c''',
            '''c = a + b;''', 'test_kernel')
        kern(self.a_huge, self.b_huge, self.c_huge)


cupy_perf.run(__name__)
