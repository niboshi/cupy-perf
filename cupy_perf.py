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

        ts = numpy.empty((n,), dtype=numpy.float64)
        ev = self._ev

        for i in range(n_warmup):
            func(self)

        for i in range(n):
            ev.synchronize()
            t1 = time.perf_counter()

            func(self)

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
