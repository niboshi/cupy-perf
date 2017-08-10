import inspect
import sys
import time

import cupy
import numpy


class PerfCaseResult(object):
    def __init__(self, name, ts):
        self.name = name
        self.ts = ts

    def __str__(self):
        return '{:<20s}: {:.03f} us   +/- {:.03f} us'.format(
            self.name,
            self.ts.mean() * 1e6,
            self.ts.std() * 1e6)


class PerfCases(object):
    def __init__(self):
        self._ev = cupy.cuda.stream.Event()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def get_cases(self):
        prefix = 'perf_'
        cases = []
        for name in dir(self):
            if name.startswith(prefix):
                f = getattr(self, name)
                if callable(f):
                    _, linum = inspect.getsourcelines(f)
                    name = name[len(prefix):]
                    cases.append((linum, name, f))

        cases = sorted(cases)
        for linum, name, f in cases:
            yield name, f

    def run(self):
        cases = list(self.get_cases())
        for case_name, case in cases:
            self.setUp()
            result = self._run_perf(case_name, case)
            self.tearDown()
            print(str(result))

    def _run_perf(self, name, func, n=10000, n_warmup=10):
        ts = numpy.empty((n,), dtype=numpy.float64)
        ev = self._ev

        for i in range(n_warmup):
            func()

        for i in range(n):
            ev.synchronize()
            t1 = time.perf_counter()

            func()

            t2 = time.perf_counter()
            ev.synchronize()
            ts[i] = t2 - t1

        return PerfCaseResult(name, ts)


def run(module_name):
    mod = sys.modules[module_name]
    classes = []
    for name, cls in inspect.getmembers(mod):
        if inspect.isclass(cls) and issubclass(cls, PerfCases):
            _, linum = inspect.getsourcelines(cls)
            classes.append((linum, cls))

    classes = sorted(classes)
    for linum, cls in classes:
        cases = cls()
        cases.run()
