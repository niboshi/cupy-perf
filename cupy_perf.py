import inspect
import sys
import time

import cupy
import numpy


class PerfCase:
    def __init__(self, func):
        self.func = func
        self.n = 10000
        self.n_warmup = 10


def attr(**kwargs):
    def decorator(case):
        if isinstance(case, PerfCase):
            case_ = case
        else:
            case_ = PerfCase(case)

        for key, val in kwargs.items():
            setattr(case_, key, val)
        return case_
    return decorator


class PerfCaseResult(object):
    def __init__(self, name, ts):
        self.name = name
        self.ts = ts

    def cpu_mean(self):
        return self.ts[0].mean()

    def cpu_std(self):
        return self.ts[0].std()

    def gpu_mean(self):
        return self.ts[1].mean()

    def gpu_std(self):
        return self.ts[1].std()

    def __str__(self):
        return '{:<20s}: {:.03f} us   +/- {:.03f} us      {:.03f} us   +/- {:.03f} us'.format(
            self.name,
            self.cpu_mean() * 1e6,
            self.cpu_std() * 1e6,
            self.gpu_mean() * 1e6,
            self.gpu_std() * 1e6)


class PerfCases(object):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def get_cases(self):
        prefix = 'perf_'
        cases = []
        for name in dir(self):
            if name.startswith(prefix):
                obj = getattr(self, name)
                if isinstance(obj, PerfCase):
                    case = obj
                    func = obj.func
                elif callable(obj):
                    case = PerfCase(obj.__func__)
                    func = obj
                else:
                    continue

                name = name[len(prefix):]
                _, linum = inspect.getsourcelines(func)
                cases.append((linum, name, case))

        cases = sorted(cases)
        for linum, name, f in cases:
            yield name, f

    def run(self):
        cases = list(self.get_cases())
        for case_name, case in cases:
            self.setUp()

            if isinstance(case, PerfCase):
                pass
            else:
                case = PerfCase(case)
            result = self._run_perf(case_name, case)
            self.tearDown()
            print(str(result))

    def _run_perf(self, name, case):
        func = case.func
        n = case.n
        n_warmup = case.n_warmup

        ts = numpy.empty((2, n,), dtype=numpy.float64)
        ev1 = cupy.cuda.stream.Event()
        ev2 = cupy.cuda.stream.Event()

        for i in range(n_warmup):
            func(self)

        for i in range(n):
            ev1.record()
            t1 = time.perf_counter()

            func(self)

            t2 = time.perf_counter()
            ev2.record()
            ev2.synchronize()
            ts[0, i] = t2 - t1
            ts[1, i] = cupy.cuda.get_elapsed_time(ev1, ev2) * 1e-3

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
